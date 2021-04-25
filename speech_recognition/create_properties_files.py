# this script automatically creates properties files
names = ["bio_1l2m5h_mni2", "bio_1l2m5h_mni3", "bio_1l2m5h_mni4", "bio_1l2m5h_mni5", "bio_1l2m5h_tal3", "bio_1l2m5h_tal4", "bio_1l2m5h_tal5", "bio_1l2m5h_tal6", "bio_1l2m5h_tal7", "bio_1l2m5h_tal8", "bio_1l2m5h_tal9", "bio_1l2m5h_tal10"]
lifs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
dir = "C:/Users/Anne/Documents/Studium/PhD/python scripts/"
text = ""
with open(dir + "modified.properties", "r") as f:
    text = f.readlines();

for name in names:
    name2 = name[11:14] + "_by_" + name[14:]  # gives tal_by_10
    size = int(name[14:])
    text[0] = "projectName=" + name2[:3].upper() + name2[3:] + "_jackson\n"  # gives TAL_by_5
    text[1] = "stdDirectory=C:\\\\Users\\\\anwendt\\\\Documents\\\\Python Scripts\\\\" + name + "\\\\jackson\\\\\n"
    text[6] = "mappingCoordinatesFileName=" + name2 + ".csv\n"
    text[7] = "inputCoordinatesFileName=for_" + name2 + ".csv\n"
    text[8] = "smallWorldRadius={0:.1f}\n".format(size * 2.5)

    for lif in lifs:
        text[3] = "lifThresholdVoltage={0:.2f}\n".format(lif)
        lif_text = "{0:03d}".format(int(lif * 100))
        with open(dir + name + "/lif_" + lif_text + ".properties", "w") as f:
            f.writelines(text)



# loop over template names: mni_by_2, mni_by_3 etc
# loop over different LIF settings
# write top bit

#0 projectName=MNI_by_3_jackson  # this one should be easy
#1 stdDirectory=C:\\Users\\anwendt\\Documents\\Python Scripts\\bio_1l2m5h_mni3\\jackson\\  # for this one just replace one part of the string
#2 coreName=LIF  # stays the same because Israel said SLIF doesn't work anyways
#3 lifThresholdVoltage=0.01  # depends on loop parameter
#4 slifThresholdVoltage=0.5  # stays the same
#5 izBehaviour=H  # stays the same
#6 mappingCoordinatesFileName=mni_by_3.csv  # this one should be easy
#7 inputCoordinatesFileName=for_mni_by_3.csv  # this one should be easy
#8 smallWorldRadius=7.5  # this one is important

# then copy rest of the file
# save it using the template name and the loop parameter name
