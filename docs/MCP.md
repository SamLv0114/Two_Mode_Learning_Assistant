# 本地 MCP 工具服务

`src/agents/mcp_server.py` 将现有 `ToolRegistry` 的两个只读工具
`search_knowledge_base` 和 `fetch_full_paper` 暴露为标准 MCP 工具，使用
官方 Python SDK 的 stdio 传输。服务每次工具调用都重新验证 access token
和用户状态；refresh token、缺失 token、停用账号会被拒绝。它不修改
`ToolRegistry`，也不在公网开启未经配置的 HTTP 入口。

启动前设置 `LEARNING_ASSISTANT_ACCESS_TOKEN` 为当前登录用户的 access token，
并确保 MCP 进程与 API 使用同一个固定 `SECRET_KEY`，
然后由 MCP 客户端启动：

```bash
python -m src.agents.mcp_server
```

客户端通过 MCP 发现工具及参数 schema；实际执行、超时和熔断仍由
`ToolRegistry` 负责。项目原有 Function Calling 是模型选择应用内工具的机制；
MCP 在客户端与工具服务之间提供发现和调用协议。

互操作验证：`python -m unittest discover -s tests -p 'test_mcp_server.py' -v`。
当前只提供本地单用户进程演示；远程多用户 HTTP MCP 需要独立的 OAuth
和请求级用户上下文设计。
