import datetime

# 마스터 폴더위치
MASTER_FOLDER = r"C:\Users\yjshi\Documents\Research\pwd_v3\data\\"
TD_FOLDER = MASTER_FOLDER + "tensor_decompo\\"
print("Master Folder : ",MASTER_FOLDER)
print("TD_FOLDER : ",TD_FOLDER)

# 현재시각 타임스탬프
t = datetime.datetime.now()
TIME = "_%dM%dD_%dh%dm%ds" % (t.month,t.day,t.hour,t.minute,t.second)
print("Time : ",TIME)



#########################
# 1.8characters.py
# 2.generate_strength.py
# 3.combine_strength.py

