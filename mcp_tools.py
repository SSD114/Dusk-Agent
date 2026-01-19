from mcp.server.fastmcp import FastMCP
import platform
import subprocess

# 初始化 MCP 服务
mcp = FastMCP("DuskTools")

# 1. 天气工具
@mcp.tool()
def get_weather(city: str) -> str:
    """
    [必须使用此工具查天气]
    用于获取任何城市或地区的实时天气情况。
    当用户询问"天气"、"气温"、"下雨"时，必须且只能调用此工具。
    
    Args:
        city: 城市名称 (例如: "北京", "Shanghai", "Tokyo")
    """
    print(f"--- [MCP Server] 正在执行天气查询: {city} ---")
    
    # 模拟真实数据返回
    return f"{{ 'city': '{city}', 'temp': '18°C', 'condition': '阴有小雨', 'tips': '出门记得带伞' }}"

# 2. 乘法工具
@mcp.tool()
def multiply(a: float, b: float) -> float:
    """
    [数学计算工具]
    仅用于计算两个数字的乘积。严禁用于其他用途
    """
    print(f"--- [MCP Server] 正在执行乘法: {a} * {b} ---")
    return a * b


# --- 辅助函数：用于执行系统命令 ---
def run_command(command):
    """运行系统shell命令并返回清理后的结果"""
    try:
        # 使用 subprocess 执行命令
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
        # 解码并去除首尾空白
        return result.decode('utf-8', errors='ignore').strip()
    except Exception:
        return None
    
@mcp.tool()
def check_user_hardware() -> str:
    """
    [硬件规格检测]
    获取当前电脑的核心硬件型号信息，包括CPU型号、主板型号和显卡型号。
    当用户询问"我的电脑配置"、"显卡是什么"、"硬件参数"时调用此工具。
    不要用于查询使用率(CPU占比等)，只用于查询型号名称。
    """
    print(f"--- [MCP Server] 正在执行硬件扫描 ---")
    
    os_type = platform.system()
    cpu_model = "未知设备"
    mobo_model = "未知设备"
    gpu_model = "未知设备"

    try:
        # --- Windows 系统 ---
        if os_type == "Windows":
            # 1. CPU
            raw_cpu = run_command("wmic cpu get name")
            if raw_cpu:
                # wmic 输出通常包含表头和多行空行，取第二行有效数据
                lines = [line.strip() for line in raw_cpu.split('\n') if line.strip()]
                if len(lines) > 1: cpu_model = lines[1]

            # 2. 主板 (制造商 + 产品名)
            raw_mobo = run_command("wmic baseboard get manufacturer, product")
            if raw_mobo:
                lines = [line.strip() for line in raw_mobo.split('\n') if line.strip()]
                if len(lines) > 1: mobo_model = lines[1]

            # 3. 显卡
            raw_gpu = run_command("wmic path win32_videocontroller get name")
            if raw_gpu:
                # 显卡可能有多个，过滤掉表头
                gpus = [line.strip() for line in raw_gpu.split('\n') if line.strip() and "Name" not in line]
                gpu_model = ", ".join(gpus)

        # --- macOS 系统 ---
        elif os_type == "Darwin":
            cpu_model = run_command("sysctl -n machdep.cpu.brand_string")
            mobo_model = run_command("sysctl -n hw.model") # Mac通常返回型号ID (如 MacBookPro18,3)
            # 获取显卡 (system_profiler较慢，但准确)
            raw_gpu = run_command("system_profiler SPDisplaysDataType | grep 'Chipset Model'")
            if raw_gpu:
                 gpu_model = raw_gpu.split(':')[-1].strip()

        # --- Linux 系统 ---
        elif os_type == "Linux":
            # CPU
            raw_cpu = run_command("grep 'model name' /proc/cpuinfo | head -n 1")
            if raw_cpu: cpu_model = raw_cpu.split(':')[-1].strip()
            
            # 主板
            mobo_vendor = run_command("cat /sys/class/dmi/id/board_vendor")
            mobo_name = run_command("cat /sys/class/dmi/id/board_name")
            if mobo_vendor or mobo_name:
                mobo_model = f"{mobo_vendor or ''} {mobo_name or ''}".strip()
            
            # GPU
            raw_gpu = run_command("lspci | grep -i vga")
            if raw_gpu:
                gpu_model = raw_gpu.split(':')[-1].strip()

        return (
            f"硬件规格报告:\n"
            f"- 操作系统: {os_type}\n"
            f"- CPU型号: {cpu_model}\n"
            f"- 主板型号: {mobo_model}\n"
            f"- 显卡型号: {gpu_model}"
        )

    except Exception as e:
        return f"硬件检测发生错误: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
