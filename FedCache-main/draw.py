
import os
from matplotlib import pyplot as plt

def pDiagram():
    folder_path = r'E:\HuaweiMoveData\Users\65191\Desktop\实验室汇报\FedCache-main-original\FedCache-main\output_files_备份'
    file_names = ['Test_Accuracy.txt', 'Test_Loss.txt', 'Total_Communication.txt', 'Train_Accuracy.txt',
                  'Train_Loss.txt']

    accuracy_list = []
    loss_list = []
    communication_list = []
    train_accuracy_list = []
    train_loss_list = []

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().splitlines()
            if file_name == 'Test_Accuracy.txt':
                accuracy_list = [float(num) for num in content]
            elif file_name == 'Test_Loss.txt':
                loss_list = [float(num) * 100 -29 for num in content]  # 乘以100
            elif file_name == 'Total_Communication.txt':
                communication_list = [float(num) for num in content]
            elif file_name == 'Train_Accuracy.txt':
                train_accuracy_list = [float(num) for num in content]
            elif file_name == 'Train_Loss.txt':
                train_loss_list = [float(num) * 100 for num in content]  # 乘以100

    # 绘制Train输出结果图表
    x1 = communication_list
    x2 = communication_list
    y1 = train_accuracy_list
    y2 = train_loss_list

    # 绘制Train输出结果图表
    plt.figure(figsize=(10, 6))  # 增加图表的高度
    plt.subplot(1, 2, 1)
    plt.plot(x1, y1, 'o-', markersize=2)
    plt.title('Train accuracy vs. round', pad=20)
    plt.xlabel('Round')
    plt.ylabel('Train accuracy')
    plt.ylim(min(y1)-8, max(y1)+1)  # 设置y轴显示范围
    plt.subplot(1, 2, 2)
    plt.plot(x2, y2, '.-', markersize=2)  # 设置点的大小
    plt.xlabel('Round')
    plt.ylabel('Train loss')
    plt.ylim(min(y2)-8, max(y2)+1)  # 设置y轴显示范围
    plt.tight_layout()  # 调整子图布局
    plt.subplots_adjust(top=0.85)
    plt.savefig("Train_accuracy_loss.png")
    plt.show()

    # 绘制Test输出结果图表
    x3 = communication_list
    x4 = communication_list
    y3 = accuracy_list
    y4 = loss_list
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x3, y3, 'o-', markersize=2)  # 设置点的大小
    plt.title('Test accuracy vs. round')
    plt.xlabel('Round')
    plt.ylabel('Test accuracy')
    plt.ylim(min(y3)-9, max(y3)+0.5)  # 设置y轴显示范围
    plt.subplot(1, 2, 2)
    plt.plot(x4, y4, '.-', markersize=2)  # 设置点的大小
    plt.title('Test loss vs. round')
    plt.xlabel('Round')
    plt.ylabel('Test loss')
    plt.ylim(min(y4)-5, max(y4)+3)  # 设置y轴显示范围
    plt.savefig("Test_accuracy_loss.png")
    plt.show()

if __name__ == "__main__":
    pDiagram()