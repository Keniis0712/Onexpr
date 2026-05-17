"""FastAPI demo. Covers async endpoints, pydantic models, dependency
injection, path/query/body params, and exception handling.

Run the original:
    python examples/fastapi_demo.py
or onexpr-obfuscated:
    python onexpr.py --input examples/fastapi_demo.py --output obf.py
    python obf.py

Then poke at it:
    GET  http://127.0.0.1:8000/items?limit=2
    GET  http://127.0.0.1:8000/items/1
    POST http://127.0.0.1:8000/items     {"name":"hat","price":9.99}
    GET  http://127.0.0.1:8000/error
    GET  http://127.0.0.1:8000/stream
"""

import asyncio
from typing import Annotated

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


class Item(BaseModel):
    id: int | None = None
    name: str = Field(..., min_length=1)
    price: float = Field(..., gt=0)
    tags: list[str] = []


class Store:
    """In-memory store; mirrors what a real DB layer would look like."""

    def __init__(self) -> None:
        self._items: dict[int, Item] = {}
        self._next_id = 1

    def list(self, limit: int) -> list[Item]:
        return list(self._items.values())[:limit]

    def get(self, item_id: int) -> Item:
        if item_id not in self._items:
            raise HTTPException(status_code=404, detail=f"item {item_id} not found")
        return self._items[item_id]

    def add(self, item: Item) -> Item:
        item.id = self._next_id
        self._next_id += 1
        self._items[item.id] = item
        return item


# Singleton store instance, injected via Depends.
_store = Store()
_store.add(Item(name="seed-a", price=1.0, tags=["seed"]))
_store.add(Item(name="seed-b", price=2.5))


def get_store() -> Store:
    return _store


app = FastAPI(title="onexpr fastapi demo")


@app.get("/")
async def root() -> dict:
    return {"ok": True, "msg": "see /items, /items/{id}, /stream, /error"}


@app.get("/items")
async def list_items(
    store: Annotated[Store, Depends(get_store)],
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> list[Item]:
    return store.list(limit)


@app.get("/items/{item_id}")
async def get_item(
    item_id: int,
    store: Annotated[Store, Depends(get_store)],
) -> Item:
    return store.get(item_id)


@app.post("/items")
async def create_item(
    item: Item,
    store: Annotated[Store, Depends(get_store)],
) -> Item:
    return store.add(item)


@app.get("/stream")
async def stream():
    """Async-generator response — exercises async generators end-to-end."""

    async def gen():
        for i in range(5):
            await asyncio.sleep(0.05)
            yield f"chunk-{i}\n"

    return StreamingResponse(gen(), media_type="text/plain")


@app.get("/error")
async def trigger_error():
    raise HTTPException(status_code=418, detail="i'm a teapot")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
