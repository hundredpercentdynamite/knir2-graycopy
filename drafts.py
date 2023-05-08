# LUT
# table = []
# for i in range(256):
#     value = i / 255
#     x = None
#     if value < 0.04045:
#         x = value / 12.92
#     else:
#         x = pow((value + 0.055) / 1.055, 2.4)
#     table.append(x * 255)
# table = np.array(table, np.uint8)
# print("LEN", table)
# def gammaCorrection(src):
#     st = time.time()
#     image = cv2.applyColorMap(src, table)
#     end = time.time()
#     elapsed_time = end - st
#     print('Execution time LUT:', elapsed_time, 'seconds')
#     return image



def getHSPlot(hsvImage):
    h, s, v = splitChannels(hsvImage)
    figure = plt.figure(figsize=(15, 15), dpi=80)
    plot = figure.add_subplot()
    plot.set_xlabel('H')
    plot.set_ylabel('S')
    plot.scatter(x=h, y=s, marker=".", s=1)
    plt.show()
    return figure
