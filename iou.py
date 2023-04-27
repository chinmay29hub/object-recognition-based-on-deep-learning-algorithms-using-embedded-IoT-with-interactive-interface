def func(ax1:int,ay1:int,ax2:int,ay2:int,bx1:int,by1:int,bx2:int,by2:int):
    area1=(ax2-ax1)*(ay2-ay1)
    area2=(bx2-bx1)*(by2-by1)
    xOverlap= min(ax2,bx2)-max(ax1,bx1)
    yOverlap=min(ay2,by2)-max(ay1,by1)
    commonArea=xOverlap*yOverlap
    totalArea=area1+area2-commonArea
    iou=commonArea/totalArea
    return ("%s, %s, %s, %s" % (area1,area2,commonArea,iou))

# Read the input files and store the data in two lists
with open("./objects_values_yolo.txt", "r") as f1, open("./faster_predicted.txt", "r") as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()

# Iterate over the lists and call the function with the values from each line
with open("output_faster.txt", "w") as f_out:
    for i in range(len(lines1)):
        values1 = list(map(int, lines1[i].strip().split(",")))
        values2 = list(map(int, lines2[i].strip().split(",")))
        result = func(*values1, *values2)
        f_out.write(result + "\n")
