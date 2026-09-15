"""
BAPD Client: Automated interface to the Classical Art Research Centre (CARC)
Beazley Archive Pottery Database at Oxford University.
Enables querying, metadata harvesting, and high-resolution image downloads.
"""

import os
import re
import time
import urllib.parse
from typing import Dict, List, Optional, Any, Tuple
import requests
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np
import io

CARC_BASE_URL = "https://www.carc.ox.ac.uk"
SEARCH_URL = f"{CARC_BASE_URL}/XDB/ASP/searchOpen.asp"
VERIFY_URL = f"{CARC_BASE_URL}/XDB/ASP/verifySubmit.asp"


class BAPDRecord:
    """
    Structured representation of a vase record in the Beazley Archive.
    """
    def __init__(self, guid: str, fields: Dict[str, str], image_urls: List[str]):
        self.guid = guid
        self.fields = fields
        self.image_urls = image_urls

    @property
    def vase_number(self) -> str:
        return self.fields.get("Vase Number", "").strip()

    @property
    def fabric(self) -> str:
        return self.fields.get("Fabric", "").strip()

    @property
    def technique(self) -> str:
        return self.fields.get("Technique", "").strip()

    @property
    def shape(self) -> str:
        return self.fields.get("Shape Name", "").strip()

    @property
    def provenance(self) -> str:
        return self.fields.get("Provenance", "").strip()

    @property
    def date_range(self) -> str:
        return self.fields.get("Date", "").strip()

    @property
    def artist(self) -> str:
        return self.fields.get("Attributed To", "").strip()

    @property
    def decoration(self) -> str:
        return self.fields.get("Decoration", "").strip()

    @property
    def collection(self) -> str:
        return self.fields.get("Last Recorded Collection", "").strip()

    @property
    def publication(self) -> str:
        return self.fields.get("Publication Record", "").strip()

    @property
    def inscriptions(self) -> str:
        return self.fields.get("Inscriptions", "").strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "guid": self.guid,
            "vase_number": self.vase_number,
            "fabric": self.fabric,
            "technique": self.technique,
            "shape": self.shape,
            "provenance": self.provenance,
            "date_range": self.date_range,
            "artist": self.artist,
            "decoration": self.decoration,
            "collection": self.collection,
            "publication": self.publication,
            "inscriptions": self.inscriptions,
            "image_urls": self.image_urls,
            "raw_fields": self.fields
        }


class BAPDClient:
    """
    Session-aware client for searching and scraping BAPD records and images.
    """
    def __init__(self, request_timeout: int = 15):
        self.timeout = request_timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
        })
        self._verified = False

    def ensure_verified(self, force: bool = False) -> bool:
        """
        Solves CARC's browser verification challenge if not already verified.
        """
        if self._verified and not force:
            return True

        try:
            r = self.session.get(SEARCH_URL, timeout=self.timeout)
            if "verifyForm" in r.text:
                token_match = re.search(r'name=["\']token["\']\s+value=["\']([^"\']+)["\']', r.text)
                if token_match:
                    token = token_match.group(1)
                    # CARC enforces a minimum 2-second client-side delay before submitting token
                    time.sleep(2.5)
                    self.session.post(
                        VERIFY_URL,
                        data={"token": token},
                        headers={
                            "Referer": SEARCH_URL,
                            "Origin": CARC_BASE_URL,
                            "Content-Type": "application/x-www-form-urlencoded"
                        },
                        timeout=self.timeout
                    )
            self._verified = True
            return True
        except Exception as e:
            print(f"Error during BAPD verification: {e}")
            return False

    def search_vases(
        self,
        query: str,
        with_images: bool = True,
        limit: int = 25,
        start_from: int = 1
    ) -> List[Tuple[str, str]]:
        """
        Searches BAPD for a query term (e.g. 'HERAKLES', 'ATHENA', 'DIONYSOS', 'APOLLO').
        Returns list of tuples: (guid, summary_text).
        """
        self.ensure_verified()
        encoded_query = urllib.parse.quote(query.strip())
        
        # Batch size for CARC
        no_to_display = min(limit, 50)
        
        url = (
            f"{SEARCH_URL}?action=showResults&search={encoded_query}"
            f"&startFrom={start_from}&noToDisplay={no_to_display}"
        )
        if with_images:
            url += "&chkImages=true&WithImages=Yes"

        try:
            res = self.session.get(url, timeout=self.timeout)
            soup = BeautifulSoup(res.text, "html.parser")
            
            records = []
            for row in soup.find_all("div", class_="searchResult"):
                onclick = row.get("onclick", "")
                guid_match = re.search(r"/record/(\{[A-Za-z0-9\-]+\})", onclick)
                if not guid_match:
                    continue
                guid = guid_match.group(1)
                
                cell = row.find("div", class_="searchResultCell")
                summary = cell.get_text(" ", strip=True) if cell else ""
                records.append((guid, summary))
                
                if len(records) >= limit:
                    break
                    
            return records
        except Exception as e:
            print(f"Error searching BAPD for '{query}': {e}")
            return []

    def get_vase_record(self, guid: str) -> Optional[BAPDRecord]:
        """
        Retrieves complete metadata and high-resolution image links for a vase GUID.
        """
        self.ensure_verified()
        record_url = f"{CARC_BASE_URL}/record/{guid}"
        
        try:
            r = self.session.get(record_url, timeout=self.timeout)
            if r.status_code != 200:
                return None
                
            soup = BeautifulSoup(r.text, "html.parser")
            
            # Extract fields from recordText <li> items
            fields = {}
            rec_text = soup.find("div", class_="recordText")
            if rec_text:
                for li in rec_text.find_all("li"):
                    txt = li.get_text(" ", strip=True)
                    if ":" in txt:
                        k, v = txt.split(":", 1)
                        fields[k.strip()] = v.strip()

            # Extract high-resolution image URLs
            image_urls = []
            
            # 1. Look for 'recordDetailsLarge.asp' links
            large_links = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "recordDetailsLarge.asp" in href:
                    full_href = urllib.parse.urljoin(CARC_BASE_URL, href)
                    if full_href not in large_links:
                        large_links.append(full_href)

            # Sort links: Prioritize photographic plates (.A, .B, Plate, CVA) and deprioritize/exclude drawings
            def link_priority(url_str: str) -> int:
                u = url_str.upper()
                if "BEIL" in u or "PROFILE" in u or "DRAWING" in u or "SCHNITT" in u:
                    return 999
                if "%2EA%2F" in u or ".A/" in u or ".A." in u:
                    return 1 # Side A
                if "%2EB%2F" in u or ".B/" in u or ".B." in u:
                    return 2 # Side B
                if "%2EI%2F" in u or ".I/" in u:
                    return 3 # Inside / Tondo
                return 10 # Other photographic plates

            large_links.sort(key=link_priority)

            for large_page_url in large_links:
                try:
                    r_large = self.session.get(large_page_url, timeout=self.timeout)
                    soup_large = BeautifulSoup(r_large.text, "html.parser")
                    for img in soup_large.find_all("img"):
                        src = img.get("src", "")
                        if "displayImage.asp" in src or "/SPIFF/" in src:
                            full_src = urllib.parse.urljoin(CARC_BASE_URL, src)
                            if full_src not in image_urls:
                                image_urls.append(full_src)
                except Exception:
                    pass

            # 2. Fallback: Check standard images directly embedded in the record page
            if not image_urls:
                for img in soup.find_all("img"):
                    src = img.get("src", "")
                    if "/Vases/SPIFF/" in src and "BEIL" not in src.upper():
                        full_src = urllib.parse.urljoin(CARC_BASE_URL, src)
                        if full_src not in image_urls:
                            image_urls.append(full_src)

            return BAPDRecord(guid=guid, fields=fields, image_urls=image_urls)
        except Exception as e:
            print(f"Error fetching BAPD record {guid}: {e}")
            return None

    def download_image(
        self,
        image_url: str,
        output_path: str,
        min_size: int = 200,
        reject_drawings: bool = True
    ) -> bool:
        """
        Downloads an image from BAPD, verifies it is a valid picture, and saves as JPEG/PNG.
        If reject_drawings=True, automatically rejects black-and-white 2D line drawings / profiles.
        """
        self.ensure_verified()
        try:
            res = self.session.get(image_url, timeout=self.timeout)
            if res.status_code != 200 or len(res.content) < 1000:
                return False
                
            img = Image.open(io.BytesIO(res.content))
            # Verify minimum dimensions
            if img.width < min_size and img.height < min_size:
                return False

            # Computer vision check: Line drawings have predominantly white scan backgrounds (>60%)
            # with very few dark pixels (<8%) and high mean intensity (>218).
            if reject_drawings:
                gray = np.array(img.convert("L"))
                white_frac = float(np.mean(gray > 235))
                dark_frac = float(np.mean(gray < 50))
                mean_val = float(np.mean(gray))
                if (white_frac > 0.58 and dark_frac < 0.08) or mean_val > 218:
                    return False
                
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            img.convert("RGB").save(output_path, format="JPEG", quality=92)
            return True
        except Exception as e:
            print(f"Error downloading {image_url} to {output_path}: {e}")
            return False
