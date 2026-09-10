"""Streamlit UI for the developer code-review agent."""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
import streamlit as st

SYSTEM_PROMPT = (
    "You are a senior developer performing a code review. Use "
    "static_code_check on the snippet, then write a short, "
    "constructive review comment covering what to fix and why."
)


@tool
def static_code_check(code: str) -> str:
    """Run a lightweight static check on a Python code snippet."""
    issues = []
    if '"""' not in code and "'''" not in code:
        issues.append("No docstring found.")
    if "TODO" in code:
        issues.append("Contains TODO comment(s) left in the code.")
    if code.count("\n") > 40:
        issues.append("Function/file may be too long — consider splitting it.")
    return "; ".join(issues) if issues else "No obvious issues found."


def build_agent(api_key: str):
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)
    return create_agent(llm, [static_code_check], system_prompt=SYSTEM_PROMPT)


def collect_sources(snippet: str, uploaded_files) -> list[tuple[str, str]]:
    sources: list[tuple[str, str]] = []
    if snippet.strip():
        sources.append(("pasted snippet", snippet))
    for uploaded in uploaded_files or []:
        text = uploaded.read().decode("utf-8", errors="replace")
        sources.append((uploaded.name, text))
    return sources


def extract_output(result: dict) -> str:
    messages = result.get("messages") or []
    if not messages:
        return "No review was returned."
    last = messages[-1]
    content = getattr(last, "content", last)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            else:
                parts.append(str(item))
        return "\n".join(parts).strip() or "No review was returned."
    return str(content).strip() or "No review was returned."


def extract_tool_steps(result: dict) -> list[tuple[str, str, str]]:
    steps: list[tuple[str, str, str]] = []
    for message in result.get("messages") or []:
        name = getattr(message, "type", "") or message.__class__.__name__
        if name in {"tool", "ToolMessage"}:
            tool_name = getattr(message, "name", "tool")
            content = getattr(message, "content", "")
            steps.append((tool_name, str(content), "observation"))
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            if isinstance(call, dict):
                steps.append(
                    (
                        str(call.get("name", "tool")),
                        str(call.get("args", "")),
                        "call",
                    )
                )
    return steps


def review_source(agent, name: str, code: str) -> dict:
    print(f"[review] starting review for: {name}")
    print(f"[review] code length: {len(code)} chars, lines: {code.count(chr(10)) + 1}")
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Review this code from {name}:\n{code}",
                }
            ]
        }
    )
    print(f"[review] finished review for: {name}")
    return result


def main() -> None:
    st.set_page_config(page_title="Developer Review Agent", page_icon="🧑‍💻")
    st.title("Developer Review Agent")
    st.caption(
        "Paste a Python snippet and/or upload `.py` files. "
        "The agent runs `static_code_check`, then writes a short review."
    )

    with st.sidebar:
        st.header("OpenAI")
        api_key = st.text_input(
            "OpenAI API key",
            type="password",
            help="Used only for this session. It is not written to disk.",
        )
        st.caption("Model: `gpt-4o-mini`")

    snippet = st.text_area(
        "Python snippet",
        height=220,
        placeholder="def process(data):\n    ...",
    )
    uploaded_files = st.file_uploader(
        "Python files to review",
        type=["py"],
        accept_multiple_files=True,
    )

    if not st.button("Run review", type="primary"):
        return

    if not api_key.strip():
        st.error("Enter an OpenAI API key in the sidebar.")
        return

    sources = collect_sources(snippet, uploaded_files)
    if not sources:
        st.error("Paste a snippet or upload at least one `.py` file.")
        return

    agent = build_agent(api_key.strip())
    for name, code in sources:
        st.subheader(name)
        with st.expander("Code submitted", expanded=False):
            st.code(code, language="python")

        with st.spinner(f"Reviewing {name}..."):
            try:
                result = review_source(agent, name, code)
            except Exception as exc:
                st.error(f"Review failed for {name}: {exc}")
                continue

        st.markdown(extract_output(result))
        steps = extract_tool_steps(result)
        if steps:
            with st.expander("Agent steps (tool calls)", expanded=False):
                for index, (tool_name, detail, kind) in enumerate(steps, start=1):
                    st.markdown(f"**Step {index}: `{tool_name}` ({kind})**")
                    st.write(detail)


if __name__ == "__main__":
    main()
