"""
为整个工程提供统一路径
"""
import os

def get_project_root():
    #获得当前项目绝对路径
    current_file = os.path.abspath(__file__)
    #获得文件所在文件夹路径
    current_dir=os.path.dirname(current_file)
    #获得工程根目录
    project_root=os.path.dirname(current_dir)
    return project_root
def get_abs_path(relative_path: str) -> str:
    project_root=get_project_root()
    return os.path.join(project_root, relative_path)

if __name__ == '__main__':
    print(get_abs_path("./config/config.txt"))