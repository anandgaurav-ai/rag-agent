from langchain_core.messages import ToolMessage
tm = ToolMessage(content='test', tool_call_id='123')
print(type(tm))
print(tm)