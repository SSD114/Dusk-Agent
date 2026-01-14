from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")



from mcp.server.fastmcp import FastMCP

# 初始化 MCP 服务
mcp = FastMCP("DuskTools")

# 1. 天气工具：描述要长，关键词要多
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

# 2. 乘法工具：保留一个数学工具作为对比，但描述要精准

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """
    [数学计算工具]
    仅用于计算两个数字的乘积。严禁用于其他用途，如查询天气。
    """
    print(f"--- [MCP Server] 正在执行乘法: {a} * {b} ---")
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
