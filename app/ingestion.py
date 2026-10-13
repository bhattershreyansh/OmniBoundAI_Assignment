import hashlib
import httpx
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.schemas import DocumentIn, Chunk
from app.errors import UnreachableURLError

_splitter: RecursiveCharacterTextSplitter | None = None

# Standard browser User-Agent to avoid immediate 403 blocks
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    )
}


def make_doc_id(text_or_url: str) -> str:
    """Stable SHA-256 hash to ensure idempotency."""
    return hashlib.sha256(text_or_url.strip().encode("utf-8")).hexdigest()[:16]


def split_text_into_chunks(text: str) -> list[str]:
    """Splits text on paragraphs, lines, sentences, then words up to 400 tokens."""
    global _splitter
    if _splitter is None:
        _splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=settings.CHUNK_SIZE_TOKENS,
            chunk_overlap=settings.CHUNK_OVERLAP_TOKENS
        )
    return _splitter.split_text(text)


async def fetch_url_text(client: httpx.AsyncClient, url: str) -> str:
    """
    Fetches URL content safely with browser User-Agent,
    guards against non-http protocols and oversized pages,
    and extracts readable text.
    """
    if not (url.startswith("http://") or url.startswith("https://")):
        raise UnreachableURLError(url, "Only http and https protocols are supported.")

    try:
        # Stream response to cap size without reading unbounded bodies into RAM.
        async with client.stream(
            "GET",
            url,
            headers=HEADERS,
            timeout=settings.FETCH_TIMEOUT_SECONDS,
            follow_redirects=True
        ) as response:
            if response.status_code != 200:
                raise UnreachableURLError(url, f"Received HTTP {response.status_code}")

            content_bytes = bytearray()
            async for chunk in response.aiter_bytes():
                content_bytes.extend(chunk)
                if len(content_bytes) > settings.FETCH_MAX_BYTES:
                    raise UnreachableURLError(
                        url,
                        f"Response exceeded maximum allowed size of {settings.FETCH_MAX_BYTES} bytes"
                    )

            html_text = content_bytes.decode(response.encoding or "utf-8", errors="replace")

        soup = BeautifulSoup(html_text, "html.parser")

        # Strip scripts, styles, forms, and navigation boilerplate
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned = "\n".join(lines)
        if not cleaned:
            raise UnreachableURLError(url, "Extracted empty text from page.")
        return cleaned

    except httpx.RequestError as exc:
        raise UnreachableURLError(url, f"Network connection error: {str(exc)}")
    except UnreachableURLError:
        raise
    except Exception as exc:
        raise UnreachableURLError(url, f"Parsing failure: {str(exc)}")


async def process_document_to_chunks(
    tenant_id: str,
    doc: DocumentIn,
    client: httpx.AsyncClient
) -> tuple[str, list[Chunk]]:
    """Extracts text, computes doc_id, splits into chunks, and attaches metadata."""
    if doc.url:
        raw_text = await fetch_url_text(client, doc.url)
        raw_identifier = doc.url
        source = doc.source or doc.url
    else:
        raw_text = doc.text or ""
        source = doc.source or "manual-text"
        raw_identifier = source or raw_text

    doc_id = make_doc_id(raw_identifier)
    text_chunks = split_text_into_chunks(raw_text)

    chunks: list[Chunk] = []
    for idx, chunk_text in enumerate(text_chunks):
        chunk_id = f"{doc_id}-{idx}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                tenant_id=tenant_id,
                doc_id=doc_id,
                chunk_index=idx,
                source=source,
                brand=doc.brand
            )
        )

    return doc_id, chunks
