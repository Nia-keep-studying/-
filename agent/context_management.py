from langchain.agents.middleware import ModelRequest,wrap_model_call
from langchain_core.messages.utils import count_tokens_approximately


def solit_messages_for_summary(messages):
    human_positions = []

    for index,message in enumerate(messages):
        if message.type == "human":
            human_positions.append(index)

    if len(human_positions) <= 4:
        return [],list(messages)

    recent_start = human_positions[-4]

    older_messages = messages[:recent_start]
    recent_messages = messages[recent_start:]

    return older_messages,recent_messages

@wrap_model_call
def observe_context_tokens(request:ModelRequest,handler):
    messages = list(request.messages)

    if request.system_message is not None:
        messages.insert(0,request.system_message)

    estiamted_tokens = count_tokens_approximately(
        messages,
        tools = request.tools,
        chars_per_token=2.0
    )

    print(f"本次模型输入预计 token: {estiamted_tokens}")

    return handler(request)
