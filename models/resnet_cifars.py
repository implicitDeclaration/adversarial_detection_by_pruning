import torch
import torch.nn as nn
import torch.nn.functional as F


conv_num_cfg = {
    'resnet18' : 8,
    'resnet34' : 16,
    'resnet50' : 16,
    'resnet101' : 33,
    'resnet152' : 50 
}

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, honey, index, stride=1):
        super(BasicBlock, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        middle_planes = int(planes * honey[index] / 10)

        self.conv1 = nn.Conv2d(inplanes, int(planes * honey[index] / 10), kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(planes * honey[index] / 10))
        
        self.conv2 = nn.Conv2d(int(planes * honey[index] / 10), planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out





class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, honey, index, stride=1):
        super(Bottleneck, self).__init__()
        pr_channels = int(planes * honey[index] / 10)
        self.conv1 = nn.Conv2d(in_planes, pr_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(pr_channels)
        self.conv2 = nn.Conv2d(pr_channels , pr_channels,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(pr_channels)
        self.conv3 = nn.Conv2d(pr_channels, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, honey=None):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.honey = honey
        self.current_conv = 0

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,  bias=False)
        self.bn1 = nn.BatchNorm2d(64)
      


        self.layer1 = self._make_layers(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layers(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layers(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layers(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



    def _make_layers(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes,
                self.honey, self.current_conv, stride))
            self.current_conv +=1
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    
        # function to extact the multiple features
    def feature_list(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, out_list
    
    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        out = F.relu(self.bn1(self.conv1(x)))
        if layer_index == 1:
            out = self.layer1(out)
        elif layer_index == 2:
            out = self.layer1(out)
            out = self.layer2(out)
        elif layer_index == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
        elif layer_index == 4:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)               
        return out

    # function to extact the penultimate features
    def penultimate_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        penultimate = self.layer4(out)
        out = F.avg_pool2d(penultimate, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y, penultimate

def resnet(cfg, honey= None, num_classes = 10):
    # if honey == None:
    #     honey = conv_num_cfg[cfg] * [10]
    if cfg == 'resnet18':
        return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes, honey=honey)
    elif cfg == 'resnet34':
        return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes, honey=honey)
    elif cfg == 'resnet50':
        return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes, honey=honey)
    elif cfg == 'resnet101':
        return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes, honey=honey)
    elif cfg == 'resnet152':
        return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes, honey=honey)


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3], num_classes =10, honey = None)

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])







