import time


def get_timer():
    return time.time()


def print_device_timer(start, end):
    elapsed_time = end - start
    print(f'TIME ELAPSED: {elapsed_time} seconds\n')
    return


def timer_data(func):
    print('\n-----> START TIMER <-----\n')
    start_time = get_timer()

    # param function here
    func()

    print('\n-----> END TIMER <-----\n')
    end_time = get_timer()
    print_device_timer(start_time, end_time)
    return