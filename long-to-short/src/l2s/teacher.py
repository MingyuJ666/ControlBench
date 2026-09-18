"""Single-teacher cached requests. Transport errors never count as filter failures."""
import os
from urllib.parse import urlsplit, urlunsplit
from .storage import read_json, write_json

TEACHER = 'gpt-5.1'
DECODING = dict(reasoning_effort='none', max_completion_tokens=8192, n=1)


def api_base_url(value):
    """Gateway roots need /v1; already configured API paths are preserved."""
    parts = urlsplit(value)
    if parts.scheme not in ('http', 'https') or not parts.netloc:
        raise RuntimeError('OPENAI_BASE_URL must be an HTTP(S) endpoint')
    path = parts.path.rstrip('/') or '/v1'
    return urlunsplit((parts.scheme, parts.netloc, path + '/', parts.query, parts.fragment))


class Teacher:
    def __init__(self, directory):
        from openai import OpenAI
        self.directory = directory
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'],
                             base_url=api_base_url(os.environ['OPENAI_BASE_URL']), timeout=180, max_retries=2)

    def __call__(self, stage, attempt, prompt, temperature):
        request = dict(model=TEACHER, messages=[dict(role='user', content=prompt)],
                       temperature=temperature, **DECODING)
        path = self.directory / f'{stage}-{attempt}.json'
        if path.exists():
            return read_json(path)['reply']
        completion = self.client.chat.completions.create(**request)
        if isinstance(completion, str):
            raise RuntimeError('Teacher endpoint returned text instead of a Chat Completions JSON response')
        if len(completion.choices) != 1:
            raise RuntimeError('Teacher returned an unexpected number of choices')
        choice = completion.choices[0]
        if not choice.message.content:
            raise RuntimeError('Teacher returned empty content; retry this run after checking its channel')
        reply = dict(text=choice.message.content, finish_reason=choice.finish_reason)
        write_json(path, dict(request=request, reply=reply, response_id=completion.id,
                              returned_model=completion.model,
                              usage=completion.usage.model_dump() if completion.usage else None))
        return reply
