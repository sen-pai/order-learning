
# green gradient, darker green to white. 40 colors. [[color_rgb, rank = 0].. color_rgb, rank = 9]]
GREEN_COLORS = list(zip([
(5, 255, 23),
(10, 254, 26),
(14, 254, 30),
(19, 253, 33),
(24, 253, 37),
(29, 252, 40),
(33, 252, 44),
(38, 251, 47),
(43, 251, 51),
(48, 250, 54),
(52, 250, 58),
(57, 249, 61),
(62, 249, 65),
(67, 248, 68),
(71, 248, 72),
(76, 247, 75),
(81, 247, 79),
(86, 246, 82),
(90, 246, 86),
(95, 245, 89),
(100, 245, 93),
(105, 244, 96),
(109, 244, 100),
(114, 243, 103),
(119, 243, 107),
(124, 242, 110),
(128, 242, 114),
(133, 241, 117),
(138, 241, 121),
(143, 240, 124),
(147, 240, 128),
(152, 239, 131),
(157, 239, 135),
(162, 238, 138),
(166, 238, 142),
(171, 237, 145),
(176, 237, 149),
(181, 236, 152),
(185, 236, 156),
(190, 235, 159)
],
list(range(40))))

# background is always white
BG_COLOR = (255, 255, 255)

IMAGE_SIZE = (64, 64)


EXTREMES = {
    "upper": list(range(30, 40)),
    "lower": list(range(0, 10))
}

CENTER = {
    "upper": list(range(20, 30)),
    "lower": list(range(10, 20))
}
