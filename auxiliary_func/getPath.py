import os


def get_current_path():
    return os.getcwd().replace('\\', '/')


def get_base_path():
    og_path = get_current_path()
    l_path = og_path.split('/')
    base_path = l_path[0]
    for i in range(1, len(l_path)-1):
        base_path += f'/{l_path[i]}'
    return base_path


def get_chapter_data_path(num_part, num_chapter):
    base_path = get_base_path()
    data_path = f'{base_path}/dlwpt-code/data/p{num_part}ch{num_chapter}'
    return data_path
