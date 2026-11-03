import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. ATENCIÓN ESPACIAL (Filtro para apagar el ruido de fondo)
# ==============================================================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv(scale)
        
        return x * self.sigmoid(scale)

# ==============================================================================
# 2. ATENCIÓN DE CANALES (Extraído de Pezzotti et al.)
# ==============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

# ==============================================================================
# 3. BLOQUES RRDB (Extracción de contornos afilados y detalles estructurales)
# ==============================================================================
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_channel=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels + growth_channel, growth_channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_channel, growth_channel, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channels + 3 * growth_channel, growth_channel, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channels + 4 * growth_channel, in_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.ca = ChannelAttention(in_channels) # Integración de Channel Attention

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x5 = self.ca(x5) # Ponderación dinámica de canales
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, in_channels, growth_channel=32):
        super().__init__()
        self.rdb1 = DenseBlock(in_channels, growth_channel)
        self.rdb2 = DenseBlock(in_channels, growth_channel)
        self.rdb3 = DenseBlock(in_channels, growth_channel)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

# ==============================================================================
# 4. BLOQUES U-NET (Mitigación del ruido estocástico mediante Pooling)
# ==============================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels), 
            nn.LeakyReLU(0.2, inplace=True),
            ChannelAttention(out_channels), # Integración de Channel Attention
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            ChannelAttention(out_channels) # Integración de Channel Attention
        )

    def forward(self, x):
        return self.double_conv(x)

# ==============================================================================
# 5. RED COMPLETA DETERMINISTA (Bifurcada para Varianza y Densidad)
# ==============================================================================
class RRDB_UNet_Deterministic(nn.Module):
    def __init__(self, in_channels=1, base_dim=64, num_rrdb=3, con_hierro=False):
        super().__init__()
        self.con_hierro = con_hierro
        
        # --- ENCODER (Extracción de características compartidas) ---
        self.inc = DoubleConv(in_channels, base_dim)
        
        self.down1 = nn.Sequential(nn.AvgPool2d(2), DoubleConv(base_dim, base_dim * 2))
        self.down2 = nn.Sequential(nn.AvgPool2d(2), DoubleConv(base_dim * 2, base_dim * 4))
        self.down3 = nn.Sequential(nn.AvgPool2d(2), DoubleConv(base_dim * 4, base_dim * 8))
        
        self.spatial_attention = SpatialAttention(kernel_size=7)
        
        # --- CUELLO DE BOTELLA (Procesamiento topológico profundo) ---
        self.bottleneck = nn.Sequential(
            *[RRDB(in_channels=base_dim * 8, growth_channel=base_dim * 4) for _ in range(num_rrdb)]
        )
        
        # --- DECODER (Reconstrucción topológica compartida) ---
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
                                 nn.Conv2d(base_dim * 8, base_dim * 4, kernel_size=3, padding=1))
        self.conv_up3 = DoubleConv(base_dim * 8, base_dim * 4)
        
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
                                 nn.Conv2d(base_dim * 4, base_dim * 2, kernel_size=3, padding=1))
        self.conv_up2 = DoubleConv(base_dim * 4, base_dim * 2)
        
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
                                 nn.Conv2d(base_dim * 2, base_dim, kernel_size=3, padding=1))
        self.conv_up1 = DoubleConv(base_dim * 2, base_dim)
        
        # --- CABEZALES DE SALIDA BIFURCADOS ---
        # Cabezal 1: Predicción de Varianza Angular 
        self.head_variance = nn.Sequential(
            nn.Conv2d(base_dim, base_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_dim // 2, 1, kernel_size=1)
        )
        
        # Cabezal 2: Predicción de Integral de Radiación 1/X0 
        self.head_density = nn.Sequential(
            nn.Conv2d(base_dim, base_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_dim // 2, 1, kernel_size=1)
        )

        # Cabezal 3 (opcional, V14): probabilidad de hierro por píxel (logits). Obliga a decidir
        # "barra / no barra" con forma rígida en lugar de dibujar un sombreado tenue
        if con_hierro:
            self.head_iron = nn.Sequential(
                nn.Conv2d(base_dim, base_dim // 2, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_dim // 2, 1, kernel_size=1)
            )

    def forward(self, x):
        # Bajada
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Atención y Cuello de Botella
        x4_attended = self.spatial_attention(x4)
        bot = self.bottleneck(x4_attended)
        
        # Subida
        u3 = self.up3(bot)
        u3 = torch.cat([u3, x3], dim=1) 
        u3 = self.conv_up3(u3)
        
        u2 = self.up2(u3)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv_up2(u2)
        
        u1 = self.up1(u2)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.conv_up1(u1)
        
        # Bifurcación final
        out_variance = self.head_variance(u1)
        out_density = self.head_density(u1)

        if self.con_hierro:
            return out_variance, out_density, self.head_iron(u1)
        return out_variance, out_density