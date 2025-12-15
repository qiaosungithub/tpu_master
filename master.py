import time
import subprocess
import logging
from multiprocessing import Pool

# ================= 配置区域 =================

CHECK_INTERVAL = 300 
MAX_WORKERS = 10
DRY_RUN = False

# default Service Account
DEFAULT_SA = "373438850578-compute@developer.gserviceaccount.com"

ZONES = [
    "us-central1-a",
    "us-central1-b",
    "us-central2-b",
    "us-east1-d",
    "us-east5-a",
    "us-east5-b",
    "europe-west4-a",
    "asia-northeast1-b",
]

REGION_SA_MAP = {
    "us-central1": "bucket-us-central1@he-vision-group.iam.gserviceaccount.com",
    "us-central2": "bucket-us-central2@he-vision-group.iam.gserviceaccount.com",
    "us-east1": "373438850578-compute@developer.gserviceaccount.com",
    "us-east5": "bucket-us-east5@he-vision-group.iam.gserviceaccount.com",
    "asia-northeast1": "bucket-asia@he-vision-group.iam.gserviceaccount.com",
    "europe-west4": "373438850578-compute@developer.gserviceaccount.com"
}

# ===========================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("tpu_enforcer.log"), logging.StreamHandler()]
)

def get_region_from_zone(zone):
    return "-".join(zone.split("/")[-1].split("-")[:-1])

def delete_tpu(name, zone):
    if 'kangyang' in name:
        logging.info(f"\033[032m宽限 zhh 的 TPU: {name}\033[0m")
        return
    
    if 'gzy' in name:
        logging.info(f"\033[032m宽限 zhengyang 的 TPU: {name}\033[0m")
        return
    
    if name in ['kmh-tpuvm-v6e-64-spot-103', 'kmh-tpuvm-v6e-64-spot-109', 'kmh-tpuvm-v6e-64-spot-52']:
        logging.info(f"\033[032m宽限 xibo 的 TPU: {name}\033[0m")
        return
    
    if name in ['kmh-tpuvm-v6e-64-spot-108']:
        logging.info(f"\033[032m宽限 sqa 的 TPU: {name}\033[0m")
        return
    
    if 'kmh-tpuvm-v3-8' in name or 'kmh-tpuvm-v4-8-' in name:
        logging.warning(f"\033[032m[Warning] found problematic dev machine: {name}\033[0m")
        return

    if DRY_RUN:
        logging.warning(f"🚫 [DRY RUN] 发现违规，拟删除 TPU: {name}")
        return
    
    logging.warning(f"🧨 正在强制执行删除: {name} in {zone}")
    cmd = ["gcloud", "compute", "tpus", "tpu-vm", "delete", name, "--zone", zone, "--quiet", "--async"]
    subprocess.run(cmd)

def check_single_tpu(tpu):
    # tpu: a dict, have keys like 'name', 'zone', etc.
    print(f"Checking TPU: {tpu} in zone {tpu.get('zone')}")
    name = tpu.get('name').split('/')[-1]
    zone = tpu.get('zone').split('/')[-1]
    region = get_region_from_zone(zone)
    
    expected_sa = REGION_SA_MAP.get(region)
    if not expected_sa:
        return f"[{name}] 跳过：未定义 Region {region} 的合规策略"

    # 逻辑：查找占用 TPU 的用户，并获取其 active 账号（支持环境变量和 gcloud auth 两种方式）
    remote_command = (
        "PID=$(sudo lsof -t /dev/accel* /dev/vfio/* 2>/dev/null | head -n 1); "
        # if not PID found, return "CHECK_RES:IDLE"
        "if [ -z \"$PID\" ]; then echo \"CHECK_RES:IDLE\"; exit 0; fi; "
        # 获取对应的 Linux 用户
        "TPU_USER=$(ps -o user= -p \"$PID\"); "
        
        # check environ GOOGLE_APPLICATION_CREDENTIALS
        "KEY_PATH=$(sudo strings /proc/$PID/environ | grep '^GOOGLE_APPLICATION_CREDENTIALS=' | cut -d= -f2-); "
        "ENV_SA=''; "
        "if [ -n \"$KEY_PATH\" ] && [ -f \"$KEY_PATH\" ]; then "
        "  ENV_SA=$(grep -oP '\"client_email\":\\s*\"\\K[^\"]+' \"$KEY_PATH\" 2>/dev/null || echo ''); "
        "fi; "
        
        # check gcloud auth
        "ALL_ACTIVE=$(sudo -u \"$TPU_USER\" gcloud auth list --filter='status:ACTIVE' --format='value(account)'); "
        "ACCOUNT_COUNT=$(echo \"$ALL_ACTIVE\" | grep -v '^$' | wc -l); "
        "GCLOUD_SA=''; "
        "if [ \"$ACCOUNT_COUNT\" -eq 1 ]; then "
        "  GCLOUD_SA=$(echo \"$ALL_ACTIVE\" | head -n 1); "
        "fi; "
        
        "echo \"CHECK_RES:BUSY|USER:$TPU_USER|ENV_SA:$ENV_SA|GCLOUD_SA:$GCLOUD_SA|GCLOUD_COUNT:$ACCOUNT_COUNT\""
    )

    # 构造 gcloud ssh 命令
    # 使用 --worker=all 确保多机节点（如 v3-32）的每个 worker 都受检
    ssh_cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "ssh", name,
        "--zone", zone,
        "--worker=all",
        "--ssh-flag=-n",
        "--command", remote_command
    ]

    try:
        # gcloud ssh 可能会比较慢，超时设为 30 秒
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=40)
        
        if result.returncode != 0:
            return f"[{name}] SSH 失败: {result.stderr.strip()}"

        # 解析输出
        # 因为用了 --worker=all，输出会有多行（每个 worker 一行或多行）
        lines = result.stdout.strip().split('\n')
        
        # 步骤1：解析所有行，收集检测结果
        check_results = []
        for line in lines:
            if "CHECK_RES:" not in line:
                raise ValueError(f"[{name}] no CHECK_RES: in line: {line}. All output: {result.stdout.strip()}")
                
            payload = line.split("CHECK_RES:")[1].strip()
            
            if payload == "IDLE":
                continue
            
            # 解析 BUSY|USER:xxx|ENV_SA:yyy|GCLOUD_SA:zzz|GCLOUD_COUNT:n
            parts = payload.split('|')
            user_info = {p.split(':', 1)[0]: p.split(':', 1)[1] if ':' in p else '' for p in parts if ':' in p}
            check_results.append(user_info)
        
        # 如果所有 worker 都是 IDLE，机器空闲
        if not check_results:
            logging.info(f"TPU [{name}] is idle.")
            return f"TPU [{name}] is idle."
        
        # 步骤2：收集所有发现的账号并去重
        all_users = set()
        all_env_sas = set()
        all_gcloud_sas = set()
        all_gcloud_counts = []
        
        for result_item in check_results:
            tpu_user = result_item.get('USER', '')
            env_sa = result_item.get('ENV_SA', '')
            gcloud_sa = result_item.get('GCLOUD_SA', '')
            gcloud_count = result_item.get('GCLOUD_COUNT', '0')
            
            if tpu_user:
                all_users.add(tpu_user)
            if env_sa:
                all_env_sas.add(env_sa)
            if gcloud_sa:
                all_gcloud_sas.add(gcloud_sa)
            if gcloud_count:
                all_gcloud_counts.append(int(gcloud_count))
        
        # 合并所有发现的 Service Account（去重后）
        all_found_sas = set()
        all_found_sas.update(all_env_sas)
        all_found_sas.update(all_gcloud_sas)
        
        logging.info(f"TPU [{name}] found {len(check_results)} workers, user: {all_users}, "
                    f"environ SA: {all_env_sas}, gcloud SA: {all_gcloud_sas}")
        
        # 步骤3：统一检查合规性
        # 检查1：必须至少有一个认证方式
        if not all_found_sas:
            logging.error(f"TPU [{name}] 🚨 违规！用户: {all_users}, 未检测到任何认证方式")
            delete_tpu(name, zone)
            return f"TPU [{name}] 判定违规（未检测到认证）并触发删除操作"
        
        # 检查2：如果任何 worker 有多个 gcloud 账号，违规
        if any(count > 1 for count in all_gcloud_counts):
            max_count = max(all_gcloud_counts)
            logging.error(f"TPU [{name}] 🚨 违规！用户: {all_users}, 检测到多个 gcloud ACTIVE 账号（最多: {max_count}）")
            delete_tpu(name, zone)
            return f"TPU [{name}] 判定违规（多个 gcloud 账号）并触发删除操作"
        
        # 检查3：验证所有发现的 SA 是否合法（必须是默认账号或 region 账号）
        unauthorized_sas = [sa for sa in all_found_sas if sa != DEFAULT_SA and sa != expected_sa]
        if unauthorized_sas:
            logging.error(f"TPU [{name}] 🚨 违规！用户: {all_users}, 使用未授权账号: {unauthorized_sas}, "
                         f"允许的账号: {DEFAULT_SA} 或 {expected_sa}")
            delete_tpu(name, zone)
            return f"TPU [{name}] 判定违规（使用未授权账号: {unauthorized_sas}）并触发删除操作"
        
        # 所有检查通过
        logging.info(f"TPU [{name}] ✅ 合规：运行中，所有 Service Account 正确")
        return f"TPU [{name}] 合规：运行中，且 Service Account 正确"
    except subprocess.TimeoutExpired:
        logging.warning(f"\033[032m[Warning] [{name}] SSH 连接超时!\033[0m")
        return f"[{name}] 错误：SSH 连接超时"
    except Exception as e:
        raise ValueError(f"[{name}] 异常：{str(e)}")
        return f"[{name}] 异常：{str(e)}"

def run_audit():
    logging.info("=== 开始全量 TPU 合规性审计 ===")
    
    # 获取所有 Zone 的 TPU 列表（只获取名称和状态，过滤掉异常状态）
    all_tpus = []
    for zone in ZONES:
        list_cmd = ["gcloud", "compute", "tpus", "tpu-vm", "list", "--zone", zone, "--format=value(name,state)"]
        try:
            tpu_output = subprocess.check_output(list_cmd, text=True, stderr=subprocess.DEVNULL)
            lines = [line.strip() for line in tpu_output.strip().split('\n') if line.strip()]
            
            active_count = 0
            skipped_count = 0
            
            for line in lines:
                parts = line.split('\t')
                if len(parts) >= 2:
                    tpu_name = parts[0].strip()
                    tpu_state = parts[1].strip()
                    
                    if tpu_state in ['PREEMPTED', 'TERMINATED', 'CREATING', 'DELETING', 'REPAIRING', 'STOPPED']:
                        # logging.info(f"ignore {zone}/{tpu_name} (状态: {tpu_state})")
                        skipped_count += 1
                        continue

                    if tpu_name == 'kmh-tpuvm-v4-8-4':
                        logging.info(f"ignore {zone}/{tpu_name} (测试忽略名单)")
                        skipped_count += 1
                        continue
                    
                    all_tpus.append({
                        'name': tpu_name,
                        'zone': zone
                    })
                    active_count += 1
            
            if active_count > 0:
                logging.info(f"在 {zone} 发现 {active_count} 个活跃 TPU (跳过 {skipped_count} 个异常状态)")
                
        except subprocess.CalledProcessError:
            raise ValueError(f"无法访问 Zone: {zone}")
        except Exception as e:
            raise ValueError(f"获取 Zone {zone} 的 TPU 列表时出错: {str(e)}")

    if not all_tpus:
        logging.info("当前没有运行中的 TPU")
        return

    # 使用进程池并行检查
    with Pool(MAX_WORKERS) as p:
        summary = p.map(check_single_tpu, all_tpus)

    logging.info("=" * 10 + " 本轮审计总结 " + "=" * 10)
    for item in summary:
        if 'idle' in item.lower(): continue
        logging.info(item)

if __name__ == "__main__":
    while True:
        start_time = time.time()
        run_audit()
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        if elapsed_time > CHECK_INTERVAL:
            logging.error(f"❌ 审计运行时间 {elapsed_time:.2f} 秒超过了配置的间隔 {CHECK_INTERVAL} 秒！")
            logging.error("⚠️  请增加 CHECK_INTERVAL 或减少 MAX_WORKERS 以确保审计能在间隔时间内完成")
        else:
            wait_time = CHECK_INTERVAL - elapsed_time
            logging.info(f"审计完成，耗时 {elapsed_time:.2f} 秒。等待 {wait_time:.2f} 秒进行下一轮...")
            time.sleep(wait_time)