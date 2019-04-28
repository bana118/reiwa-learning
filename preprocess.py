import ic_module as ic
import os.path as op

i = 0
for filename in ic.FileNames :
    # ディレクトリ名入力
    #while True :
    #    dirname = input(">>「" + ic.ClassNames[i] + "」の画像のあるディレクトリ ： ")
    #    if op.isdir(dirname) :
    #        break
    #    print(">> そのディレクトリは存在しません！")
    
    # 関数実行
    dirname = "data/"+str(i+1)
    ic.PreProcess(dirname, filename, var_amount=3)
    i += 1

#filename = "img1.npy"
#dirname = "teacher/"+str(84)+"-all"
#ic.PreProcess(dirname, filename, var_amount=3)
#filename = "img2.npy"
#dirname = "teacher/"+str(87)+"-all"
#ic.PreProcess(dirname, filename, var_amount=3)
#filename = "img3.npy"
#dirname = "teacher/"+str(88)+"-all"
#ic.PreProcess(dirname, filename, var_amount=3)