import os
import datetime

LOCK_TIME_FMT = "%Y-%m-%d_%H-%M-%S"
RED, GREEN, YELLOW, PURPLE, NC = "\033[1;31m", "\033[1;32m", "\033[1;33m", "\033[1;34m", "\033[0m"
GOOD, INFO, WARNING, FAIL = f"{GREEN}[GOOD]{NC}", f"{PURPLE}[INFO]{NC}", f"{YELLOW}[WARNING]{NC}", f"{RED}[FAIL]{NC}"
def get_lock_time_str():
    """
    锁文件用时间字符串：UTC，格式 YYYY-MM-DD_HH-MM-SS（无空格、统一下划线）。
    """
    return datetime.datetime.utcnow().strftime(LOCK_TIME_FMT)

def _parse_lock_filename(filename):
    '''
    解析锁文件名 {user}_{vm_name}_{YYYY-MM-DD_HH-MM-SS}。
    返回 (user, vm_name, time_str) 或 None（格式不对时）。
    '''
    parts = filename.split('_')
    if len(parts) < 4:
        return None
    # 时间为最后两段: YYYY-MM-DD, HH-MM-SS
    time_str = f"{parts[-2]}_{parts[-1]}"
    user = parts[0]
    vm_name = '_'.join(parts[1:-2])
    return user, vm_name, time_str

def parse_lock_time_str(s):
    """
    解析 get_lock_time_str 返回的字符串为 datetime（UTC）。
    """
    return datetime.datetime.strptime(s, LOCK_TIME_FMT)

def lock_time_seconds_between(time_str_earlier, time_str_later):
    """
    计算两个锁格式时间字符串之间的间隔秒数（time_str_later - time_str_earlier）。
    返回浮点数秒，若 later 更早则返回负数。
    """
    t1 = parse_lock_time_str(time_str_earlier)
    t2 = parse_lock_time_str(time_str_later)
    return (t2 - t1).total_seconds()

def zhan(user, vm_name):
    '''
    write a file under /kmh-nfs-ssd-us-mount/code/qiao/tpu_lock,
    with name {USER_VMNAME_TIME}. Time format: YYYY-MM-DD_HH-MM-SS (no space, underscores).
    do not support when the vm_name is an alias.
    '''

    time_str = get_lock_time_str()
    user_vm_name = f"{user}_{vm_name}_{time_str}"
    lock_dir = "/kmh-nfs-ssd-us-mount/code/qiao/tpu_lock"
    with open(f"{lock_dir}/{user_vm_name}", 'w') as f:
        f.write(f"{user}_{vm_name}_{time_str}")
    print(f'{GOOD} 成功创建了叫做{user_vm_name}的占卡锁')


def check_reserved_user(tpu):
    '''
    Check the path /kmh-nfs-ssd-us-mount/code/qiao/tpu_lock, and see whether there is a file
    named {USER_VMNAME_TIME} within 30 minutes. If a file is older than 30 minutes, delete it.
    Return the reserved user name, otherwise return None.
    Time format: YYYY-MM-DD_HH-MM-SS (no space, underscores).
    '''
    now = get_lock_time_str()
    lock_dir = "/kmh-nfs-ssd-us-mount/code/qiao/tpu_lock"
    for file in os.listdir(lock_dir):
        parsed = _parse_lock_filename(file)
        if parsed is None:
            continue
        user, vm_name, time_str = parsed
        if vm_name != tpu:
            continue
        try:
            seconds_ago = lock_time_seconds_between(time_str, now)
        except (ValueError, TypeError):
            # 旧格式或非法时间，视为过期并删除
            try:
                os.remove(f"{lock_dir}/{file}")
            except OSError:
                pass
            continue
        if seconds_ago > 30 * 60:
            try:
                os.remove(f"{lock_dir}/{file}")
            except OSError:
                pass
            continue
        return user
    return None