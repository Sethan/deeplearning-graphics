import torch


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        image_size = cfg.INPUT.IMAGE_SIZE
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS
        self.num_filters = [32,64]
        
        
        self.feature_extractor38 = torch.nn.Sequential(
        #part 1 38x38
        torch.nn.Conv2d(
            in_channels=image_channels,
            out_channels=self.num_filters[0],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.num_filters[0]),
        torch.nn.MaxPool2d(2, stride=2),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.05),
        torch.nn.Conv2d(
            in_channels=self.num_filters[0],
            out_channels=self.num_filters[1],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        
        torch.nn.BatchNorm2d(self.num_filters[1]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.06),
        torch.nn.Conv2d(
            in_channels=self.num_filters[1],
            out_channels=self.num_filters[1],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.num_filters[1]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.07),
        torch.nn.Conv2d(
            in_channels=self.num_filters[1],
            out_channels=self.num_filters[1],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.num_filters[1]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.08),
        torch.nn.Conv2d(
            in_channels=self.num_filters[1],
            out_channels=self.num_filters[1],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.num_filters[1]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.09),
        torch.nn.Conv2d(
            in_channels=self.num_filters[1],
            out_channels=self.num_filters[1],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.num_filters[1]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.01),
        torch.nn.Conv2d(
            in_channels=self.num_filters[1],
            out_channels=self.num_filters[1],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.num_filters[1]),
        torch.nn.MaxPool2d(2, stride=2),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.11),
        torch.nn.Conv2d(
            in_channels=self.num_filters[1],
            out_channels=self.num_filters[1],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.num_filters[1]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.12),
        torch.nn.Conv2d(
            in_channels=self.num_filters[1],
            out_channels=self.num_filters[1],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.num_filters[1]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.13),
        torch.nn.Conv2d(
            in_channels=self.num_filters[1],
            out_channels=self.num_filters[1],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.num_filters[1]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.14),
        torch.nn.Conv2d(
            in_channels=self.num_filters[1],
            out_channels=self.output_channels[0],
            kernel_size=3,
            stride=2,
            padding=1
        )
        )
        
        self.feature_extractor19 = torch.nn.Sequential(
         
        #part 2 19x19
        torch.nn.BatchNorm2d(self.output_channels[0]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.15),
        torch.nn.Conv2d(
            in_channels=self.output_channels[0],
            out_channels=self.output_channels[0],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[0]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.16),
        torch.nn.Conv2d(
            in_channels=self.output_channels[0],
            out_channels=self.output_channels[0],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[0]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.17),
        torch.nn.Conv2d(
            in_channels=self.output_channels[0],
            out_channels=self.output_channels[0],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[0]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.18),
        torch.nn.Conv2d(
            in_channels=self.output_channels[0],
            out_channels=self.output_channels[0],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[0]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.19),
        torch.nn.Conv2d(
            in_channels=self.output_channels[0],
            out_channels=self.output_channels[0],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[0]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.2),
        torch.nn.Conv2d(
            in_channels=self.output_channels[0],
            out_channels=self.output_channels[0],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[0]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.21),
        torch.nn.Conv2d(
            in_channels=self.output_channels[0],
            out_channels=self.output_channels[0],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[0]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.22),
        torch.nn.Conv2d(
            in_channels=self.output_channels[0],
            out_channels=self.output_channels[1],
            kernel_size=3,
            stride=2,
            padding=1
        ))
        
        self.feature_extractor9 = torch.nn.Sequential(
        
        #part 3 10x10
        torch.nn.BatchNorm2d(self.output_channels[1]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.23),
        torch.nn.Conv2d(
            in_channels=self.output_channels[1],
            out_channels=self.output_channels[1],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[1]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.24),
        torch.nn.Conv2d(
            in_channels=self.output_channels[1],
            out_channels=self.output_channels[1],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[1]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.25),
        torch.nn.Conv2d(
            in_channels=self.output_channels[1],
            out_channels=self.output_channels[1],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[1]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.26),
        torch.nn.Conv2d(
            in_channels=self.output_channels[1],
            out_channels=self.output_channels[1],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[1]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.27),
        torch.nn.Conv2d(
            in_channels=self.output_channels[1],
            out_channels=self.output_channels[1],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[1]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.28),
        torch.nn.Conv2d(
            in_channels=self.output_channels[1],
            out_channels=self.output_channels[1],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[1]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.29),
        torch.nn.Conv2d(
            in_channels=self.output_channels[1],
            out_channels=self.output_channels[1],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[1]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.30),
        torch.nn.Conv2d(
            in_channels=self.output_channels[1],
            out_channels=self.output_channels[2],
            kernel_size=3,
            stride=2,
            padding=1
        ))
        
        self.feature_extractor5 = torch.nn.Sequential(
        #part 4 5x5
        torch.nn.BatchNorm2d(self.output_channels[2]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.31),
        torch.nn.Conv2d(
            in_channels=self.output_channels[2],
            out_channels=self.output_channels[2],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[2]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.32),
        torch.nn.Conv2d(
            in_channels=self.output_channels[2],
            out_channels=self.output_channels[2],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[2]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.33),
        torch.nn.Conv2d(
            in_channels=self.output_channels[2],
            out_channels=self.output_channels[2],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[2]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.34),
        torch.nn.Conv2d(
            in_channels=self.output_channels[2],
            out_channels=self.output_channels[2],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[2]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.35),
        torch.nn.Conv2d(
            in_channels=self.output_channels[2],
            out_channels=self.output_channels[2],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[2]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.36),
        torch.nn.Conv2d(
            in_channels=self.output_channels[2],
            out_channels=self.output_channels[2],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[2]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.37),
        torch.nn.Conv2d(
            in_channels=self.output_channels[2],
            out_channels=self.output_channels[2],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[2]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.38),
        torch.nn.Conv2d(
            in_channels=self.output_channels[2],
            out_channels=self.output_channels[3],
            kernel_size=3,
            stride=2,
            padding=1
        ))
        
        self.feature_extractor3 = torch.nn.Sequential(
        
        #part 5 3x3
        torch.nn.BatchNorm2d(self.output_channels[3]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.39),
        torch.nn.Conv2d(
            in_channels=self.output_channels[3],
            out_channels=self.output_channels[3],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[3]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.40),
        torch.nn.Conv2d(
            in_channels=self.output_channels[3],
            out_channels=self.output_channels[3],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.41),
        torch.nn.Conv2d(
            in_channels=self.output_channels[3],
            out_channels=self.output_channels[3],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[3]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.42),
        torch.nn.Conv2d(
            in_channels=self.output_channels[3],
            out_channels=self.output_channels[3],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[3]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.43),
        torch.nn.Conv2d(
            in_channels=self.output_channels[3],
            out_channels=self.output_channels[3],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[3]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.44),
        torch.nn.Conv2d(
            in_channels=self.output_channels[3],
            out_channels=self.output_channels[3],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[3]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.45),
        torch.nn.Conv2d(
            in_channels=self.output_channels[3],
            out_channels=self.output_channels[3],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[3]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.46),
        torch.nn.Conv2d(
            in_channels=self.output_channels[3],
            out_channels=self.output_channels[4],
            kernel_size=3,
            stride=2,
            padding=1
        ))
        
        self.feature_extractor1 = torch.nn.Sequential(
         
        #part 6 1x1
        torch.nn.BatchNorm2d(self.output_channels[4]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.48),
        torch.nn.Conv2d(
            in_channels=self.output_channels[4],
            out_channels=self.output_channels[4],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[4]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.49),
        torch.nn.Conv2d(
            in_channels=self.output_channels[4],
            out_channels=self.output_channels[4],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[4]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.50),
        torch.nn.Conv2d(
            in_channels=self.output_channels[4],
            out_channels=self.output_channels[4],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[4]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.51),
        torch.nn.Conv2d(
            in_channels=self.output_channels[4],
            out_channels=self.output_channels[4],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[4]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.52),
        torch.nn.Conv2d(
            in_channels=self.output_channels[4],
            out_channels=self.output_channels[4],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[4]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.53),
        torch.nn.Conv2d(
            in_channels=self.output_channels[4],
            out_channels=self.output_channels[4],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[4]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.54),
        torch.nn.Conv2d(
            in_channels=self.output_channels[4],
            out_channels=self.output_channels[4],
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(self.output_channels[4]),
        torch.nn.ELU(),
        torch.nn.Dropout2d(0.55),
        torch.nn.Conv2d(
            in_channels=self.output_channels[4],
            out_channels=self.output_channels[5],
            kernel_size=3,
            stride=1,
            padding=0
        ))
    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        
        out_features = []
        out = self.feature_extractor38(x)
        out_features.append(out)
        out = self.feature_extractor19(out)
        out_features.append(out)
        out = self.feature_extractor9(out)
        out_features.append(out)
        out = self.feature_extractor5(out)
        out_features.append(out)
        out = self.feature_extractor3(out)
        out_features.append(out)
        out = self.feature_extractor1(out)
        out_features.append(out)
        feature_list = [38,19,10,5,3,1]
        for idx, feature in enumerate(out_features):
            expected_shape = (self.output_channels[idx], feature_list[idx], feature_list[idx])
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

