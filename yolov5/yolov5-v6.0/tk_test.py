import tkinter
from tkinter import ttk
import cv2
from PIL import Image, ImageTk  # 图像控件


def detect(cap):
    cap.open(0)
    while cap.isOpened():
        ret, frame = cap.read()  # BGR
        frame = cv2.flip(frame, 1)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilImage = Image.fromarray(img)  # 将array转化成Image
        tkImage = ImageTk.PhotoImage(image=pilImage)  # 一个与tkinter兼容的照片图像
        canvas.create_image(0, 0, anchor='nw', image=tkImage)
        use_time = 70.1234567890
        text = f'{use_time} ms'
        time = tkinter.Label(root_win, text=text)
        time.place(x=740, y=450)
        root_win.update()
        root_win.after(1)

        if cv2.waitKey(1) == ord('q'):  # q to quit
            cap.release()
            break


def close(cap):
    cap.release()
    cv2.destroyAllWindows()


def root_quit():
    root_win.destroy()


root_win = tkinter.Tk()  # 创建主窗口
root_win.title(string="烤箱环境下的食材检测系统")
root_win.geometry('960x600')  # 设置窗口大小
# 设置主窗口的背景颜色,颜色值可以是英文单词，或者颜色值的16进制数,除此之外还可以使用tkinter内置的颜色常量
# root_win["background"] = ""
camera = cv2.VideoCapture(0)
w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 创建画布来实时显示检测结果
canvas = tkinter.Canvas(root_win, bg='white', width=w, height=h)
canvas.place(x=10, y=10)  # 画布放置位置
# 显示检测时间
label_time = tkinter.Label(root_win, text="时间 :")
label_time.place(x=700, y=450)
# 创建表格中文显示检测结果
"""
Treeview 组件是 ttk 模块的组件之一
它既可以作为树结构使用，也可以作为表格展示数据(tkinter 并没有表格控件)
"""
columns = ("class", "number")
# 设置表格高度，"headings"表示将tree用作表格
tree = ttk.Treeview(root_win, height=18, show="headings", columns=columns)
# 设置每一列的表头宽度，内容居中
tree.column("class", width=100, anchor='center')
tree.column("number", width=100, anchor='center')
# 显示表头
tree.heading('class', text="食材种类")
tree.heading('number', text="数量")
tree.place(x=700, y=10)

# 烹饪温度和时间设置
"""
通过用户点击按钮的行为来执行回调函数，是 Button 控件的主要功用。
首先自定义一个函数或者方法，然后将函数与按钮关联起来，最后，当用户按下这个按钮时，Tkinter 就会自动调用相关函数。

"""
button_start = ttk.Button(root_win, text="开始检测", state='normal', command=lambda: detect(camera))
button_start.place(x=180, y=520)
button_exit = ttk.Button(root_win, text="停止检测", state='normal', command=lambda: close(camera))
button_exit.place(x=450, y=520)
button_quit = ttk.Button(root_win, text="关闭", state='normal', command=root_quit)
button_quit.place(x=700, y=520)

root_win.mainloop()


