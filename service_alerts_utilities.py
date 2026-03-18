# ═══════════════════════════════════════════════════════════════════════════════
# REQUIRED IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
import os
import re
import time
import hashlib
import unicodedata
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — interactive strategy selector
#
# Three loaders are available:
#
#   1. database  — Supabase PostgreSQL (fastest; requires supabase_password
#                  Colab secret and a live DB connection)
#
#   2. parquet   — cloned GitHub repo parquet files parsed via GTFSFeedLoader
#                  (offline-capable; requires the repo to be cloned at
#                   /content/-detect-and-classify-traffic-disruptions-in-real-time-)
#
#   3. gtfs_loader — full GTFSDataLoader pipeline (RepositoryManager clone +
#                    GTFSFeedLoader + parse_alerts_to_dataframe); use when you
#                    need the enriched DataFrame with all active periods,
#                    image_url, direction_id, and image fields populated
#
# The cell prompts once at startup and executes only the chosen path.
# If the choice fails it prints a clear error and stops — there is no silent
# automatic fallback so you always know exactly which source was used.
# ═══════════════════════════════════════════════════════════════════════════════
# geoparse_to_dataframe
# SECTION 1: CORE UTILITY CLASSES

class Timer:
    """Context manager for timing code execution."""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.duration = time.time() - self.start
        print(f"  Time taken: {self.duration:.2f}s")
        return False


class TextProcessor:
    """Handle text processing and extraction operations."""

    @staticmethod
    def get_first_translation_text(field):
        if field and hasattr(field, 'translation') and len(field.translation) > 0:
            return field.translation[0].text
        return None

    @staticmethod
    def extract_route_from_text(text):
        if not text:
            return None
        patterns = [
            r"lijn\s*(\d+)", r"line\s*(\d+)", r"route\s*(\d+)",
            r"bus\s*(\d+)",  r"tram\s*(\d+)", r"metro\s*(\d+)",
            r"ferry\s*(\d+)",r"rail\s*(\d+)",  r"subway\s*(\d+)",
            r"(\d+)\s*line",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def clean_text(text):
        if not text:
            return ''
        text    = unicodedata.normalize('NFC', str(text))
        cleaned = re.sub(r'[^À-ÿ\w\séèêëàäâçîïôöûü]', ' ', text.lower())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned


class TimestampConverter:
    """Handle timestamp conversions and formatting."""

    @staticmethod
    def convert_timestamp_to_local_str(timestamp, timezone_str='Europe/Amsterdam'):
        if timestamp is None or (not isinstance(timestamp, pd.Timestamp) and pd.isna(timestamp)):
            return None
        try:
            if isinstance(timestamp, pd.Timestamp):
                dt = timestamp
                if dt.tzinfo is None:
                    dt = dt.tz_localize('UTC')
            else:
                dt = pd.to_datetime(timestamp, unit='s', utc=True)
            dt = dt.tz_convert(timezone_str).tz_localize(None)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            print(f"  Error converting timestamp {timestamp}: {e}")
            return None

    @staticmethod
    def extract_id_components(id_str):
        if not id_str:
            return None, None
        parts = id_str.split(':')
        if len(parts) < 3:
            return None, id_str
        date_part    = parts[2]
        rt_id_parts  = parts[:2] + parts[3:]
        rt_id        = ':'.join(rt_id_parts) if rt_id_parts else None
        return date_part, rt_id

    @staticmethod
    def convert_id_date_to_local_str(id_date_str, timezone='Europe/Amsterdam'):
        if id_date_str is None:
            return None
        try:
            timestamp = int(id_date_str)
            return TimestampConverter.convert_timestamp_to_local_str(timestamp, timezone)
        except (ValueError, TypeError):
            try:
                dt = pd.to_datetime(id_date_str, errors='coerce')
                if pd.notna(dt):
                    if dt.tzinfo is None:
                        dt = dt.tz_localize('UTC')
                    dt = dt.tz_convert(timezone).tz_localize(None)
                    return dt.strftime('%Y-%m-%d %H:%M:%S')
                return None
            except Exception:
                return None

    @staticmethod
    def convert_active_period_range(start_ts, end_ts, timezone_str='Europe/Amsterdam'):
        start_str = TimestampConverter.convert_timestamp_to_local_str(start_ts, timezone_str)
        end_str   = TimestampConverter.convert_timestamp_to_local_str(end_ts,   timezone_str)
        valid = True
        if start_str and end_str:
            try:
                s = pd.to_datetime(start_str)
                e = pd.to_datetime(end_str)
                if s >= e:
                    print(f"  Warning: active_period start >= end ({start_str} >= {end_str})")
                    valid = False
            except Exception:
                valid = False
        return {'start_str': start_str, 'end_str': end_str, 'valid': valid}


# SECTION 2: FEED PARSING  (GTFSFeedLoader retained for parquet-feed parsing;
#            RepositoryManager removed — repo scanning no longer needed)

class GTFSFeedLoader:
    """Load and parse GTFS-RT feeds."""

    VALID_GTFS_RT_VERSIONS = {'1.0', '2.0'}
    MAX_ALERT_AGE_SECONDS  = 600

    @staticmethod
    def load_gtfs_feed(parquet_path):
        try:
            from google.transit import gtfs_realtime_pb2
            df = pd.read_parquet(parquet_path)
            if df.empty or 'feed_data' not in df.columns:
                return None
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(df['feed_data'].iloc[0])
            version = getattr(feed.header, 'gtfs_realtime_version', None)
            if version and version not in GTFSFeedLoader.VALID_GTFS_RT_VERSIONS:
                print(f"  Warning: unexpected gtfs_realtime_version='{version}'")
            feed_ts = getattr(feed.header, 'timestamp', None)
            if feed_ts:
                age = time.time() - feed_ts
                if age > GTFSFeedLoader.MAX_ALERT_AGE_SECONDS:
                    print(f"  Warning: feed is {age/60:.1f} min old")
            return feed
        except Exception as e:
            print(f"✗ Error loading {parquet_path}: {e}")
            return None

    @staticmethod
    def parse_alerts_to_dataframe(feed_dict):
        from google.transit import gtfs_realtime_pb2
        alerts_list         = []
        timestamp_converter = TimestampConverter()
        text_processor      = TextProcessor()

        for filename, feed in feed_dict.items():
            feed_timestamp = getattr(feed.header, 'timestamp', None)
            if feed_timestamp is None:
                print(f"  Skipping {filename}: No timestamp in header")
                continue
            feed_time = timestamp_converter.convert_timestamp_to_local_str(feed_timestamp)

            for entity in getattr(feed, 'entity', []):
                if not entity.HasField('alert'):
                    continue
                alert      = entity.alert
                alert_info = {
                    'feed_timestamp':      feed_time,
                    'alert_id':            getattr(entity, 'id', None),
                    'cause_id':            alert.cause,
                    'cause':               gtfs_realtime_pb2.Alert.Cause.Name(alert.cause),
                    'effect_id':           alert.effect,
                    'effect':              gtfs_realtime_pb2.Alert.Effect.Name(alert.effect),
                    'description_text':    text_processor.get_first_translation_text(
                                               getattr(alert, 'description_text', None)),
                    'header_text':         text_processor.get_first_translation_text(
                                               getattr(alert, 'header_text', None)),
                    'url_text':            text_processor.get_first_translation_text(
                                               getattr(alert, 'url', None)),
                    'image_url':           text_processor.get_first_translation_text(
                                               getattr(alert, 'image', None)),
                    'informed_entities':   [],
                    'active_periods':      [],
                    'active_period_start': None,
                    'active_period_end':   None,
                }

                active_periods = getattr(alert, 'active_period', [])
                all_periods    = []
                for i, period in enumerate(active_periods):
                    period_range = timestamp_converter.convert_active_period_range(
                        period.start if hasattr(period, 'start') and period.start else None,
                        period.end   if hasattr(period, 'end')   and period.end   else None,
                    )
                    all_periods.append(period_range)
                    if i == 0:
                        alert_info['active_period_start'] = period_range['start_str']
                        alert_info['active_period_end']   = period_range['end_str']
                alert_info['active_periods'] = all_periods

                for informed_entity in getattr(alert, 'informed_entity', []):
                    entity_details = {
                        'agency_id':  getattr(informed_entity, 'agency_id',  None),
                        'route_id':   getattr(informed_entity, 'route_id',   None),
                        'route_type': getattr(informed_entity, 'route_type', None),
                        'stop_id':    getattr(informed_entity, 'stop_id',    None),
                    }
                    if informed_entity.HasField('trip'):
                        trip         = informed_entity.trip
                        trip_details = {}
                        for attr in ('trip_id','start_time','start_date',
                                     'schedule_relationship','direction_id'):
                            if trip.HasField(attr):
                                trip_details[attr] = getattr(trip, attr)
                        if trip_details:
                            entity_details['trip'] = trip_details
                    alert_info['informed_entities'].append(entity_details)

                alerts_list.append(alert_info)

        if not alerts_list:
            print(" No alerts found in feeds")
            return pd.DataFrame()

        result = pd.DataFrame(alerts_list)
        print(f"✓ Parsed {len(result):,} alerts from {len(feed_dict)} feeds")
        return result



# SECTION 2: FEED PARSING  (GTFSFeedLoader retained for parquet-feed parsing;
#            RepositoryManager removed — repo scanning no longer needed)


class GTFSFeedLoader:
    """Load and parse GTFS-RT feeds."""

    VALID_GTFS_RT_VERSIONS = {'1.0', '2.0'}
    MAX_ALERT_AGE_SECONDS  = 600

    @staticmethod
    def load_gtfs_feed(parquet_path):
        try:
            from google.transit import gtfs_realtime_pb2
            df = pd.read_parquet(parquet_path)
            if df.empty or 'feed_data' not in df.columns:
                return None
            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(df['feed_data'].iloc[0])
            version = getattr(feed.header, 'gtfs_realtime_version', None)
            if version and version not in GTFSFeedLoader.VALID_GTFS_RT_VERSIONS:
                print(f"  Warning: unexpected gtfs_realtime_version='{version}'")
            feed_ts = getattr(feed.header, 'timestamp', None)
            if feed_ts:
                age = time.time() - feed_ts
                if age > GTFSFeedLoader.MAX_ALERT_AGE_SECONDS:
                    print(f"  Warning: feed is {age/60:.1f} min old")
            return feed
        except Exception as e:
            print(f"✗ Error loading {parquet_path}: {e}")
            return None

    @staticmethod
    def parse_alerts_to_dataframe(feed_dict):
        from google.transit import gtfs_realtime_pb2
        alerts_list         = []
        timestamp_converter = TimestampConverter()
        text_processor      = TextProcessor()

        for filename, feed in feed_dict.items():
            feed_timestamp = getattr(feed.header, 'timestamp', None)
            if feed_timestamp is None:
                print(f"  Skipping {filename}: No timestamp in header")
                continue
            feed_time = timestamp_converter.convert_timestamp_to_local_str(feed_timestamp)

            for entity in getattr(feed, 'entity', []):
                if not entity.HasField('alert'):
                    continue
                alert      = entity.alert
                alert_info = {
                    'feed_timestamp':      feed_time,
                    'alert_id':            getattr(entity, 'id', None),
                    'cause_id':            alert.cause,
                    'cause':               gtfs_realtime_pb2.Alert.Cause.Name(alert.cause),
                    'effect_id':           alert.effect,
                    'effect':              gtfs_realtime_pb2.Alert.Effect.Name(alert.effect),
                    'description_text':    text_processor.get_first_translation_text(
                                               getattr(alert, 'description_text', None)),
                    'header_text':         text_processor.get_first_translation_text(
                                               getattr(alert, 'header_text', None)),
                    'url_text':            text_processor.get_first_translation_text(
                                               getattr(alert, 'url', None)),
                    'image_url':           text_processor.get_first_translation_text(
                                               getattr(alert, 'image', None)),
                    'informed_entities':   [],
                    'active_periods':      [],
                    'active_period_start': None,
                    'active_period_end':   None,
                }

                active_periods = getattr(alert, 'active_period', [])
                all_periods    = []
                for i, period in enumerate(active_periods):
                    period_range = timestamp_converter.convert_active_period_range(
                        period.start if hasattr(period, 'start') and period.start else None,
                        period.end   if hasattr(period, 'end')   and period.end   else None,
                    )
                    all_periods.append(period_range)
                    if i == 0:
                        alert_info['active_period_start'] = period_range['start_str']
                        alert_info['active_period_end']   = period_range['end_str']
                alert_info['active_periods'] = all_periods

                for informed_entity in getattr(alert, 'informed_entity', []):
                    entity_details = {
                        'agency_id':  getattr(informed_entity, 'agency_id',  None),
                        'route_id':   getattr(informed_entity, 'route_id',   None),
                        'route_type': getattr(informed_entity, 'route_type', None),
                        'stop_id':    getattr(informed_entity, 'stop_id',    None),
                    }
                    if informed_entity.HasField('trip'):
                        trip         = informed_entity.trip
                        trip_details = {}
                        for attr in ('trip_id','start_time','start_date',
                                     'schedule_relationship','direction_id'):
                            if trip.HasField(attr):
                                trip_details[attr] = getattr(trip, attr)
                        if trip_details:
                            entity_details['trip'] = trip_details
                    alert_info['informed_entities'].append(entity_details)

                alerts_list.append(alert_info)

        if not alerts_list:
            print(" No alerts found in feeds")
            return pd.DataFrame()

        result = pd.DataFrame(alerts_list)
        print(f"✓ Parsed {len(result):,} alerts from {len(feed_dict)} feeds")
        return result

class RepositoryManager:
    """Manage GitHub repository operations."""

    def __init__(self, repo_url=None, repo_name=None):
        self.repo_url  = repo_url  or \
            "https://github.com/sharonowino/-detect-and-classify-traffic-disruptions-in-real-time-"
        self.repo_name = repo_name or "-detect-and-classify-traffic-disruptions-in-real-time-"

    def setup_colab_environment(self):
        is_colab = self._mount_drive()
        self._ensure_git_installed(is_colab)
        return self._clone_repository(is_colab)

    def _mount_drive(self):
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("✓ Google Drive mounted")
            return True
        except ImportError:
            print("  Not running in Google Colab")
            return False

    def _ensure_git_installed(self, is_colab):
        import subprocess, platform
        try:
            subprocess.run(['git', '--version'], check=True, capture_output=True)
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        if is_colab:
            os.system('apt-get update && apt-get install -y git > /dev/null 2>&1')
        else:
            system = platform.system()
            if system == "Linux":
                subprocess.run(['sudo', 'apt-get', 'install', '-y', 'git'], check=True)
            elif system == "Darwin":
                subprocess.run(['brew', 'install', 'git'], check=True)
            else:
                print(f"Please install Git manually for {system}")

    def _clone_repository(self, is_colab):
        """Clone repo — raises FileNotFoundError if missing post-clone.

        REMOVAL: No longer silences git output; stderr is captured and
        logged on failure to aid debugging.
        """
        if os.path.exists(self.repo_name):
            return os.path.join(os.getcwd(), self.repo_name)

        import subprocess
        if is_colab:
            result = subprocess.run(
                f'git clone {self.repo_url}',
                shell=True, capture_output=True, text=True
            )
        else:
            result = subprocess.run(
                ['git', 'clone', self.repo_url],
                capture_output=True, text=True
            )

        if result.returncode != 0:
            print(f"  Git clone stderr: {result.stderr.strip()}")

        cloned_path = os.path.join(os.getcwd(), self.repo_name)
        if not os.path.exists(cloned_path):
            raise FileNotFoundError("Repository not found after cloning")

        print(f"✓ Repository cloned: {cloned_path}")
        return cloned_path

    def find_parquet_files(self, repo_path=None):
        """Find parquet files in the repository.

        ADDITION: Also matches .parquet.gzip and .parquet.snappy variants.
        """
        search_root = repo_path or self.repo_name
        parquet_files = []
        for root, dirs, files in os.walk(search_root):
            for file in files:
                if (file.endswith('.parquet') or
                        file.endswith('.parquet.gzip') or
                        file.endswith('.parquet.snappy')):
                    parquet_files.append(os.path.join(root, file))
        parquet_files.sort()
        return parquet_files

    def get_repository_metadata(self):
        """Return commit hash and last-modified date for reproducibility.

        ADDITION: Enables tracking of which snapshot of the data was used.
        """
        import subprocess
        meta = {}
        try:
            result = subprocess.run(
                ['git', '-C', self.repo_name, 'log', '-1', '--format=%H %ai'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(' ', 1)
                meta['commit_hash'] = parts[0] if parts else None
                meta['commit_date'] = parts[1] if len(parts) > 1 else None
        except Exception as e:
            print(f"  Could not retrieve repo metadata: {e}")
        return meta


class GTFSDataLoader:
    """Main data loading pipeline for GTFS-RT alerts."""

    def __init__(self, repo_path=None):
        self.repo_manager = RepositoryManager()
        self.feed_loader  = GTFSFeedLoader()
        self.repo_path    = repo_path

    def load_gtfs_data(self):
        """Load GTFS-RT data from parquet files.

        ADDITION: Skips zero-byte files before attempting parse.
        ADDITION: load_from_url available as a separate method.
        """
        print("\n" + "=" * 80)
        print(" LOADING GTFS-RT DATA")
        print("=" * 80)

        try:
            if self.repo_path is None:
                self.repo_path = self.repo_manager.setup_colab_environment()

            parquet_files = self.repo_manager.find_parquet_files(self.repo_path)
            if not parquet_files:
                raise FileNotFoundError("No parquet files found!")

            alerts_files = [
                f for f in parquet_files
                if 'alerts' in os.path.basename(f).lower()
            ]
            if not alerts_files:
                raise FileNotFoundError("No alerts files found!")

            print(f"✓ Found {len(alerts_files)} alerts files")

            parsed_alerts_feeds = {}
            for i, file_path in enumerate(alerts_files, start=1):
                # skip zero-byte files
                if os.path.getsize(file_path) == 0:
                    print(f"[{i}/{len(alerts_files)}] Skipping (zero-byte): "
                          f"{os.path.basename(file_path)}")
                    continue

                print(f"[{i}/{len(alerts_files)}] Loading: {os.path.basename(file_path)}")
                feed = self.feed_loader.load_gtfs_feed(file_path)
                if feed and hasattr(feed, "entity"):
                    parsed_alerts_feeds[file_path] = feed
                    print(f"  ✓ Parsed ({len(feed.entity)} entities)")
                else:
                    print(f"  ✗ Failed or empty")

            if not parsed_alerts_feeds:
                raise RuntimeError("All feed files failed to parse.")

            alerts_df = self.feed_loader.parse_alerts_to_dataframe(parsed_alerts_feeds)
            if alerts_df is None or alerts_df.empty:
                raise RuntimeError("No alerts extracted from feeds.")

            print(f"✓ Loaded {len(alerts_df):,} alerts")
            return alerts_df, parsed_alerts_feeds

        except Exception as e:
            print(f" Error: {e}")
            return None, None

    def load_from_url(self, url: str, timeout: int = 30):
        """Load a live GTFS-RT feed directly from a URL endpoint.

        ADDITION: Enables real-time ingestion from a GTFS-RT HTTP endpoint.

        Parameters
        ----------
        url     : str  — GTFS-RT protobuf endpoint URL
        timeout : int  — HTTP request timeout in seconds

        Returns
        -------
        FeedMessage or None
        """
        try:
            import requests
            from google.transit import gtfs_realtime_pb2

            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            feed = gtfs_realtime_pb2.FeedMessage()
            feed.ParseFromString(response.content)
            print(f"✓ Loaded live feed from {url} ({len(feed.entity)} entities)")
            return feed
        except Exception as e:
            print(f"✗ Failed to load from URL {url}: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — Modular Strategy Selector
# ═══════════════════════════════════════════════════════════════════════════════

# Valid loading strategies
VALID_STRATEGIES = {
    '1': 'database',
    '2': 'parquet', 
    '3': 'gtfs_loader',
    'database': 'database',
    'parquet': 'parquet',
    'gtfs_loader': 'gtfs_loader'
}


def choose_loading_strategy() -> str:
    """
    Interactive prompt for choosing the data loading strategy.
    
    Returns:
        str: Selected strategy - 'database', 'parquet', or 'gtfs_loader'
    """
    print("=" * 65)
    print("  SERVICE ALERTS — choose a loading strategy")
    print("=" * 65)
    print()
    print("  [1] database    — Supabase PostgreSQL")
    print("                    Requires: supabase_password Colab secret + DB up")
    print()
    print("  [2] parquet     — cloned-repo parquet files (GTFSFeedLoader)")
    print("                    Requires: GitHub repo cloned to /content/...")
    print()
    print("  [3] gtfs_loader — full GTFSDataLoader pipeline")
    print("                    Clones repo automatically; produces enriched DataFrame")
    print("                    (active_periods, image_url, direction_id)")
    print()

    choice = ""
    while choice not in VALID_STRATEGIES:
        choice = input("  Enter choice [1 / 2 / 3]: ").strip().lower()
        if choice not in VALID_STRATEGIES:
            print(f"  ✗ '{choice}' is not a valid option — please enter 1, 2, or 3.")

    strategy = VALID_STRATEGIES[choice]
    print()
    print(f"  Selected: {strategy}")
    print("=" * 65)
    return strategy


def load_from_database() -> tuple:
    """
    Strategy 1: Load alerts from Supabase PostgreSQL database.
    
    Returns:
        tuple: (alerts_df, parsed_alerts_feeds, source_string)
    """
    print("  Connecting to Supabase …")
    alerts_df = None
    alerts_source = "failed"
    
    try:
        from sqlalchemy import create_engine, text
        from google.colab import userdata

        db_password = userdata.get('supabase_password')
        if not db_password:
            raise EnvironmentError("supabase_password secret is empty or not set.")

        _conn_str = (
            f"postgresql://postgres.diwldyolrrebrcmqbfgb:{db_password}"
            f"@aws-1-eu-north-1.pooler.supabase.com:6543/postgres"
        )
        engine = create_engine(
            _conn_str,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 10},
        )
        with engine.connect() as _conn:
            _conn.execute(text("SELECT 1"))
        print("  DB connection OK")

        alerts_df = pd.read_sql_table('service_alerts', engine)

        if alerts_df.empty:
            raise ValueError("service_alerts table is empty.")

        alerts_source = "database"
        print(f"\n  SOURCE: database")
        print(f"  Shape : {alerts_df.shape[0]:,} rows × {alerts_df.shape[1]} cols")

    except Exception as _exc:
        alerts_source = "failed"
        print(f"\n  ✗ Database load failed: {_exc}")
        print("    → Re-run the cell and choose a different strategy.")

    return alerts_df, {}, alerts_source


def load_from_parquet() -> tuple:
    """
    Strategy 2: Load alerts from cloned-repo parquet files via GTFSFeedLoader.
    
    Returns:
        tuple: (alerts_df, parsed_alerts_feeds, source_string)
    """
    print("  Loading service_alerts_*.parquet from cloned repo …")
    parsed_alerts_feeds = {}
    alerts_source = "failed"
    
    # Check multiple possible locations for the alerts directory
    possible_paths = [
        '/content/-detect-and-classify-traffic-disruptions-in-real-time-/alerts',
        '/content/-detect-and-classify-traffic-disruptions-in-real-time-/alerts',
        'alerts',  # local directory
        './alerts',
        '/content/alerts',
    ]
    
    _alerts_dir = None
    for path in possible_paths:
        if os.path.isdir(path):
            _alerts_dir = path
            print(f"  Found alerts directory: {path}")
            break
    
    if _alerts_dir is None:
        # List what directories exist to help debug
        print("  Debug - checking available directories:")
        for check_path in ['/content', '/content/-detect-and-classify-traffic-disruptions-in-real-time-', '.']:
            if os.path.exists(check_path):
                items = os.listdir(check_path)[:10]  # First 10 items
                print(f"    {check_path}: {items}")
            else:
                print(f"    {check_path}: NOT FOUND")
        
        raise FileNotFoundError(
            f"Alerts directory not found in any of: {possible_paths}\n"
            "  Clone the repo with LFS files, or use database strategy instead:\n"
            "  !git clone https://github.com/sharonowino/-detect-and-classify-traffic-disruptions-in-real-time-\n"
            "  Or use: load_service_alerts(strategy='database')"
        )

    # Now search for parquet files in the found directory
    _parquet_files = sorted([
        os.path.join(_alerts_dir, f)
        for f in os.listdir(_alerts_dir)
        if f.startswith('service_alerts_') 
        and f.endswith('.parquet')
        and not f.startswith('.')   # skip hidden/system files
    ])

    if not _parquet_files:
        raise FileNotFoundError(
            f"No service_alerts_*.parquet files found in: {_alerts_dir}"
        )

    print(f"  Found {len(_parquet_files)} parquet file(s).")

    try:
        _loader = GTFSFeedLoader()
        for _fp in _parquet_files:
            # skip broken or missing files
            if not os.path.exists(_fp):
                print(f"  Missing file: {os.path.basename(_fp)}")
                continue

            if os.path.getsize(_fp) == 0:
                print(f"  Skipping zero-byte: {os.path.basename(_fp)}")
                continue
        
            _feed = _loader.load_gtfs_feed(_fp)
            if _feed:
                parsed_alerts_feeds[_fp] = _feed

        if not parsed_alerts_feeds:
            raise RuntimeError("All parquet files failed to parse or were empty.")

        print("\n  Entities per file:")
        for _fp, _feed in parsed_alerts_feeds.items():
            _n = len(_feed.entity) if hasattr(_feed, 'entity') else 0
            print(f"    {os.path.basename(_fp)}: {_n} entities")

        alerts_source = "parquet"
        print(f"\n  SOURCE: parquet — {len(parsed_alerts_feeds)} feed(s) loaded")

    except Exception as _exc:
        alerts_source = "failed"
        print(f"\n  ✗ Parquet load failed: {_exc}")
        print("    → Re-run the cell and choose a different strategy.")

    return None, parsed_alerts_feeds, alerts_source


def load_from_gtfs_loader() -> tuple:
    """
    Strategy 3: Load alerts using full GTFSDataLoader pipeline.
    
    Returns:
        tuple: (alerts_df, parsed_alerts_feeds, source_string)
    """
    print("  Running GTFSDataLoader (clone + parse + enrich) …")
    print("  This produces an enriched DataFrame with active_periods,")
    print("  image_url, and direction_id populated.")
    print()
    
    alerts_df = None
    parsed_alerts_feeds = {}
    alerts_source = "failed"
    
    try:
        _gtfs_loader          = GTFSDataLoader()
        alerts_df, parsed_alerts_feeds = _gtfs_loader.load_gtfs_data()

        if alerts_df is None or alerts_df.empty:
            raise RuntimeError("GTFSDataLoader returned no data.")

        alerts_source = "gtfs_loader"
        print(f"\n  SOURCE: gtfs_loader")
        print(f"  Shape : {alerts_df.shape[0]:,} rows × {alerts_df.shape[1]} cols")

    except Exception as _exc:
        alerts_source = "failed"
        print(f"\n  ✗ GTFSDataLoader failed: {_exc}")
        print("    → Re-run the cell and choose a different strategy.")

    return alerts_df, parsed_alerts_feeds, alerts_source


def load_service_alerts(strategy: str = None) -> tuple:
    """
    Main orchestration function for loading service alerts.
    
    Args:
        strategy: Optional pre-selected strategy. If None, prompts user interactively.
                  Valid values: 'database', 'parquet', 'gtfs_loader'
    
    Returns:
        tuple: (alerts_df, parsed_alerts_feeds, alerts_source)
    """
    # If no strategy provided, prompt user to choose
    if strategy is None:
        strategy = choose_loading_strategy()
    elif strategy not in VALID_STRATEGIES.values():
        raise ValueError(f"Invalid strategy: {strategy}. Must be one of: {list(VALID_STRATEGIES.values())}")
    
    # Route to appropriate loader
    if strategy == "database":
        alerts_df, parsed_alerts_feeds, alerts_source = load_from_database()
    elif strategy == "parquet":
        alerts_df, parsed_alerts_feeds, alerts_source = load_from_parquet()
    elif strategy == "gtfs_loader":
        alerts_df, parsed_alerts_feeds, alerts_source = load_from_gtfs_loader()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Final state summary
    print()
    print("=" * 65)
    if alerts_source == "database":
        print(f"  ✓ alerts_df ready  │  source=database  │  shape={alerts_df.shape}")
        print("=" * 65)
        if alerts_df is not None:
            display(alerts_df.head())
    elif alerts_source == "parquet":
        print(f"  ✓ feeds ready  │  source=parquet  │  {len(parsed_alerts_feeds)} feed(s)")
        print("=" * 65)
    elif alerts_source == "gtfs_loader":
        print(f"  ✓ alerts_df ready  │  source=gtfs_loader  │  shape={alerts_df.shape}")
        print("=" * 65)
        if alerts_df is not None:
            display(alerts_df.head())
    else:
        print("  ✗ No data loaded — alerts_source = 'failed'")
        print("    Re-run this cell and choose a working strategy.")
        print("=" * 65)
        raise RuntimeError(
            "Service alerts unavailable.\n"
            f"  Strategy attempted : {strategy}\n"
            "  Fix the error above and re-run."
        )
    

# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY CODE — Kept for backward compatibility
# The old inline code is now replaced by the modular functions above:
#   • choose_loading_strategy() - interactive prompt
#   • load_from_database()      - database strategy
#   • load_from_parquet()       - parquet strategy  
#   • load_from_gtfs_loader()  - gtfs_loader strategy
#   • load_service_alerts()     - main orchestration
# ═══════════════════════════════════════════════════════════════════════════════

# Global variables for backward compatibility (deprecated - use load_service_alerts instead)
alerts_df           = None
parsed_alerts_feeds = {}
alerts_source       = None   # 'database' | 'parquet' | 'gtfs_loader' | 'failed'

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: PREPROCESSING CLASSES

class AlertsPreprocessor:
    """Preprocess and engineer features for alerts data."""

    def __init__(self):
        self.timestamp_converter = TimestampConverter()
        self.text_processor      = TextProcessor()

    def preprocess_alerts_data(self, alerts_df):
        if alerts_df is None or alerts_df.empty:
            return pd.DataFrame()

        df = alerts_df.copy()
        df = self._extract_id_components(df)
        df = self._convert_timestamps(df)
        df = self._calculate_duration(df)
        df = self._explode_informed_entities(df)
        df = self._process_text_features(df)
        # cyclical time features applied here for universal availability
        df = self._add_cyclical_time_features(df)
        print(f"✓ Preprocessed {len(df):,} records")
        return df

    def _extract_id_components(self, df):
        if 'alert_id' in df.columns:
            id_parts = df['alert_id'].apply(
                lambda x: self.timestamp_converter.extract_id_components(x) if x else (None, None)
            )
            df['id_date_part'] = id_parts.apply(lambda x: x[0])
            df['rt_id']        = id_parts.apply(lambda x: x[1])
            df['id_time']      = pd.to_datetime(df['id_date_part'], errors='coerce')
        return df

    def _convert_timestamps(self, df):
        for col in ['feed_timestamp', 'active_period_start', 'active_period_end']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df

    def _calculate_duration(self, df):
        if {'active_period_start', 'active_period_end'}.issubset(df.columns):
            df['active_period_duration'] = df['active_period_end'] - df['active_period_start']
            df['alert_duration_min'] = df['active_period_duration'].dt.total_seconds() / 60
        return df

    def _add_cyclical_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode hour and day-of-week as sin/cos pairs for ML models.

        Column contract (enforced here and respected everywhere downstream):
          day_of_week : int  (0=Monday to 6=Sunday) - used for ML / cyclical encoding
          day_name    : str  ('Monday' to 'Sunday')  - used only for display / grouping

        ADDITION: Cyclical encoding moved here from GTFSEventReconstructor
        so all downstream classes benefit from it automatically.
        """
        ref_col = next((c for c in ['active_period_start', 'feed_timestamp', 'id_time']
                        if c in df.columns and pd.api.types.is_datetime64_any_dtype(df[c])), None)
        if ref_col:
            df['hour']        = df[ref_col].dt.hour
            df['day_of_week'] = df[ref_col].dt.dayofweek          # int 0-6
            df['day_name']    = df[ref_col].dt.day_name()          # str "Monday"...
            df['hour_sin']    = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos']    = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin']     = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos']     = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['is_weekend']  = df['day_of_week'].isin([5, 6]).astype(int)
        return df

    def _explode_informed_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Explode informed_entities column into individual rows.

        REMOVAL: Replaced the fragile in-place mutation pattern
        (df.drop(df.index, inplace=True) + manual column assignment)
        with a clean return-and-reassign approach.
        """
        if 'informed_entities' not in df.columns:
            return df

        has_entities = df['informed_entities'].apply(
            lambda x: isinstance(x, list) and len(x) > 0
        )
        if not has_entities.any():
            return df

        df_exploded = df[has_entities].explode('informed_entities').reset_index(drop=True)
        df_without  = df[~has_entities].copy()

        if not df_exploded.empty:
            entity_df   = pd.json_normalize(df_exploded['informed_entities'])
            df_exploded = pd.concat(
                [df_exploded.drop(columns='informed_entities').reset_index(drop=True),
                 entity_df],
                axis=1
            )

        return pd.concat([df_without, df_exploded], ignore_index=True)

    def _process_text_features(self, df):
        df['description_text'] = df['description_text'].fillna('')
        df['header_text']      = df['header_text'].fillna('')
        df['clean_text']       = df['description_text'].apply(self.text_processor.clean_text)
        df['text_length']      = df['description_text'].str.len()
        df['word_count']       = df['clean_text'].str.split().str.len()
        df['route']            = df['description_text'].apply(
            self.text_processor.extract_route_from_text
        )
        return df


# Module-level compiled patterns and cached cleaner.
# lru_cache on an instance method includes `self` in the cache key, preventing
# GC and making cache_clear() unreliable. Moving the cache to module level
# keeps the key as text-only and fixes both issues.
_HTML_RE    = re.compile(r'<[^>]+>')
_URL_RE     = re.compile(r'http\S+|www\.\S+')
_EMAIL_RE   = re.compile(r'\S+@\S+')
_SPECIAL_RE = re.compile(r'[^À-ÿ\w\s\u1E00-\u1EFF]')
_WS_RE      = re.compile(r'\s+')

try:
    from functools import lru_cache as _lru_cache
except ImportError:
    def _lru_cache(maxsize=128):   # no-op fallback (Python < 3.2)
        def decorator(fn): return fn
        return decorator

@_lru_cache(maxsize=10000)
def _clean_text_cached(text: str) -> str:
    """LRU-cached core text cleaner (module-level; cache key is text only)."""
    if not text:
        return ''
    text = _HTML_RE.sub('', text)
    text = _URL_RE.sub('', text)
    text = _EMAIL_RE.sub('', text)
    text = _SPECIAL_RE.sub(' ', text)
    text = _WS_RE.sub(' ', text).strip()
    return text


class TextPreprocessor:
    """Optimized multilingual text preprocessing."""

    # Dutch stopwords for language-aware filtering
    DUTCH_STOPWORDS = {
        'de', 'het', 'een', 'en', 'van', 'in', 'is', 'op', 'dat', 'die',
        'te', 'zijn', 'met', 'voor', 'niet', 'er', 'maar', 'om', 'door',
        'nog', 'ook', 'naar', 'bij', 'aan', 'dit', 'tot', 'hij', 'uit',
        'kan', 'ze', 'worden', 'heeft', 'worden', 'als', 'had', 'worden',
        'we', 'of', 'werd', 'dan', 'hoe', 'hun', 'hem', 'u', 'worden'
    }

    def __init__(self):
        self.html_pattern          = _HTML_RE
        self.url_pattern           = _URL_RE
        self.email_pattern         = _EMAIL_RE
        self.special_chars_pattern = _SPECIAL_RE
        self.whitespace_pattern    = _WS_RE

    def clean_text(self, text: str) -> str:
        """Clean text by delegating to the module-level LRU-cached function."""
        if text is None:
            return ''
        try:
            if pd.isna(text):
                return ''
        except (TypeError, ValueError):
            pass
        return _clean_text_cached(str(text))

    def cache_clear(self):
        """Clear the LRU cache to free memory between pipeline runs.

        ADDITION: Prevents unbounded memory growth on long-running processes.
        """
        _clean_text_cached.cache_clear()
        print("  TextPreprocessor LRU cache cleared.")

    def clean_text_batch(self, texts: pd.Series) -> pd.Series:
        texts = texts.fillna('').astype(str)
        texts = texts.str.replace(self.html_pattern,          '', regex=True)
        texts = texts.str.replace(self.url_pattern,           '', regex=True)
        texts = texts.str.replace(self.email_pattern,         '', regex=True)
        texts = texts.str.replace(self.special_chars_pattern, ' ', regex=True)
        texts = texts.str.replace(self.whitespace_pattern,    ' ', regex=True).str.strip()
        return texts


    def preprocess_dataframe(self, df: pd.DataFrame, use_batch: bool = True,
                             remove_stopwords: bool = False,
                             stopwords: Optional[set] = None) -> pd.DataFrame:
        """Preprocess text columns in a DataFrame.

        ADDITION: Optional language-aware stopword removal via `remove_stopwords`
        flag. Defaults to Dutch stopwords; pass a custom set via `stopwords`.
        """
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()

        sw = stopwords if stopwords is not None else self.DUTCH_STOPWORDS

        def _apply_stopwords(text: str) -> str:
            if not remove_stopwords or not text:
                return text
            words = text.split()
            return ' '.join(w for w in words if w.lower() not in sw)

        if 'description_text' in df.columns:
            if use_batch:
                df['description_text_clean'] = self.clean_text_batch(df['description_text'])
            else:
                df['description_text_clean'] = df['description_text'].apply(self.clean_text)
            if remove_stopwords:
                df['description_text_clean'] = df['description_text_clean'].apply(
                    _apply_stopwords)
            df['text_length'] = df['description_text_clean'].str.len()
            df['word_count']  = df['description_text_clean'].str.split().str.len()

        if 'header_text' in df.columns:
            if use_batch:
                df['header_text_clean'] = self.clean_text_batch(df['header_text'])
            else:
                df['header_text_clean'] = df['header_text'].apply(self.clean_text)

        text_parts = []
        if 'description_text_clean' in df.columns:
            text_parts.append(df['description_text_clean'].fillna(''))
        if 'header_text_clean' in df.columns:
            text_parts.append(df['header_text_clean'].fillna(''))

        if text_parts:
            df['combined_text'] = pd.concat(text_parts, axis=1).apply(
                lambda x: ' '.join(x), axis=1
            ).str.strip()
        else:
            df['combined_text'] = ''

        df['has_text'] = df['combined_text'].str.len() > 0

        print(f"✓ Text preprocessing complete")
        print(f"  Alerts with text:    {df['has_text'].sum():,}")
        print(f"  Alerts without text: {(~df['has_text']).sum():,}")
        if df['has_text'].any():
            print(f"  Avg text length:     {df[df['has_text']]['text_length'].mean():.2f} chars")

        return df


# SECTION 4: NLP CLASSES

class LanguageDetector:
    """Language detection with LRU-bounded cache and batch support."""

    def __init__(self, cache_max_items: int = 50000):
        try:
            from langdetect import detect, detect_langs, LangDetectException
            self._detect              = detect
            self._detect_langs        = detect_langs
            self._LangDetectException = LangDetectException
        except ImportError:
            raise ImportError("langdetect required: pip install langdetect")

        self.lang_names = {
            'nl': 'Dutch',      'en': 'English',    'fr': 'French',
            'de': 'German',     'es': 'Spanish',    'it': 'Italian',
            'pt': 'Portuguese', 'pl': 'Polish',     'ru': 'Russian',
            'ar': 'Arabic',     'zh-cn': 'Chinese', 'ja': 'Japanese',
            'ko': 'Korean',     'tr': 'Turkish',    'sv': 'Swedish',
            'da': 'Danish',     'no': 'Norwegian',  'fi': 'Finnish',
            'cs': 'Czech',      'el': 'Greek'
        }

        # use cachetools LRUCache for bounded eviction;
        # fall back to plain dict if cachetools is not installed.
        try:
            from cachetools import LRUCache
            self._cache: Dict = LRUCache(maxsize=cache_max_items)
        except ImportError:
            print("  cachetools not found — using unbounded dict cache. "
                  "Install cachetools for bounded LRU eviction.")
            self._cache = {}
        self._cache_max_items = cache_max_items

    def _cache_key(self, text: str) -> str:
        if not text:
            return "EMPTY_TEXT"
        prefix = text[:1024].encode("utf-8", errors="ignore")
        return hashlib.sha1(prefix).hexdigest()

    def detect_language(self, text: str) -> Tuple[str, float]:
        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]
        if not text or len(text.strip()) < 10:
            return 'unknown', 0.0
        try:
            langs = self._detect_langs(text)
            if langs:
                top = langs[0]
                res = (top.lang, float(top.prob))
            else:
                res = ('unknown', 0.0)
        except self._LangDetectException:
            res = ('unknown', 0.0)
        except Exception:
            res = ('unknown', 0.0)
        self._cache[key] = res
        return res

    def detect_with_fallback(self, text: str,
                              fallback_lang: str = 'nl',
                              confidence_threshold: float = 0.70) -> Tuple[str, float]:
        """Detect language with a domain-specific fallback.

        ADDITION: Falls back to `fallback_lang` (default 'nl' for Dutch
        transit data) when detection confidence is below threshold.
        """
        lang, confidence = self.detect_language(text)
        if confidence < confidence_threshold or lang == 'unknown':
            return fallback_lang, confidence
        return lang, confidence

    def detect_batch(self, texts: List[str], show_progress: bool = True,
                     n_jobs: int = 1) -> pd.DataFrame:
        """Detect languages for a list of texts.

        ADDITION: n_jobs > 1 now triggers joblib parallel processing.
        REMOVAL: n_jobs was previously accepted but silently ignored.
        """
        if n_jobs > 1:
            try:
                from joblib import Parallel, delayed
                results_raw = Parallel(n_jobs=n_jobs)(
                    delayed(self.detect_language)(t) for t in texts
                )
                results = [
                    {'language_code': lang,
                     'language_name': self.lang_names.get(lang, lang),
                     'confidence':    conf}
                    for lang, conf in results_raw
                ]
                return pd.DataFrame(results)
            except ImportError:
                print("  joblib not installed — falling back to single-threaded detection.")

        results = []
        total   = len(texts)
        for i, text in enumerate(texts):
            if show_progress and i % 100 == 0:
                print(f"  Detecting {i}/{total}...", end='\r')
            lang, confidence = self.detect_language(text)
            results.append({
                'language_code': lang,
                'language_name': self.lang_names.get(lang, lang),
                'confidence':    confidence
            })
        if show_progress:
            print(f"  Detecting {total}/{total}... Done!")
        return pd.DataFrame(results)

    def compute_language_stats(self, df: pd.DataFrame) -> Dict:
        """Pure computation of language statistics (no printing).

        ADDITION: Separated I/O from logic. Use print_language_report()
        to display results.

        REMOVAL: analyze_languages() mixed printing with stat computation;
        replaced by this method + print_language_report().
        """
        lang_stats = df['language_name'].value_counts()
        total      = len(df)
        lc = (df['confidence'] < 0.9).sum()
        hc = (df['confidence'] > 0.95).sum()
        return {
            'distribution':          lang_stats.to_dict(),
            'avg_confidence':        float(df['confidence'].mean()),
            'median_confidence':     float(df['confidence'].median()),
            'std_confidence':        float(df['confidence'].std()),
            'low_confidence_count':  int(lc),
            'high_confidence_count': int(hc),
            'total':                 total,
        }

    def print_language_report(self, stats: Dict) -> None:
        """Print a formatted language distribution report.

        ADDITION: Display logic separated from computation.
        """
        print("\n" + "=" * 80)
        print("LANGUAGE DISTRIBUTION")
        print("=" * 80)
        total = stats['total']
        print("\nDetected Languages:")
        for lang, count in stats['distribution'].items():
            pct = (count / total) * 100
            print(f"  {lang}: {count:,} ({pct:.2f}%)")
        print(f"\nConfidence Statistics:")
        print(f"  Average: {stats['avg_confidence']:.3f}")
        print(f"  Median:  {stats['median_confidence']:.3f}")
        print(f"  Std Dev: {stats['std_confidence']:.3f}")
        lc, hc = stats['low_confidence_count'], stats['high_confidence_count']
        print(f"  Low  (<0.90): {lc:,} ({lc/total*100:.2f}%)")
        print(f"  High (>0.95): {hc:,} ({hc/total*100:.2f}%)")

    def analyze_languages(self, df: pd.DataFrame) -> Dict:
        """Analyze and print language distribution (convenience wrapper)."""
        stats = self.compute_language_stats(df)
        self.print_language_report(stats)
        return stats


class MultilingualNER:
    """Multilingual NER with integrated geoparsing support."""

    def __init__(self, model_name: str = "Davlan/xlm-roberta-base-ner-hrl",
                 device: Optional[int] = None, batch_size: int = 16, max_length: int = 512,
                 enable_geocoding: bool = True, geocoder_user_agent: str = "gtfs-ner-geoparser"):

        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if max_length < 10:
            raise ValueError(f"max_length must be >= 10, got {max_length}")

        self.batch_size   = batch_size
        self.max_length   = max_length
        self.entity_types = {
            'PER':  'Person',
            'LOC':  'Location',
            'ORG':  'Organization',
            'MISC': 'Miscellaneous'
        }

        try:
            import torch
            from transformers import pipeline as hf_pipeline
            from transformers import AutoTokenizer, AutoModelForTokenClassification
        except ImportError:
            raise ImportError("Run: pip install transformers torch")

        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.device = device

        print(f"Loading NER model: {model_name} (device={'GPU' if device >= 0 else 'CPU'})...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            model     = AutoModelForTokenClassification.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}")

        self.ner_pipeline = hf_pipeline(
            task                 = "ner",
            model                = model,
            tokenizer            = tokenizer,
            aggregation_strategy = "simple",
            device               = device,
            batch_size           = batch_size
        )
        self._model_name = model_name
        print("✓ NER model loaded!")

        self._geocode_cache        = {}
        self.geocoder              = None
        self._GeocoderTimedOut     = None
        self._GeocoderServiceError = None

        if enable_geocoding:
            self._init_geocoder(geocoder_user_agent)

    #  Private helpers ─

    def _init_geocoder(self, user_agent: str) -> None:
        try:
            from geopy.geocoders import Nominatim
            from geopy.exc import GeocoderTimedOut, GeocoderServiceError
            self.geocoder              = Nominatim(user_agent=user_agent)
            self._GeocoderTimedOut     = GeocoderTimedOut
            self._GeocoderServiceError = GeocoderServiceError
            print("✓ Geocoder initialised (Nominatim)")
        except ImportError:
            print(" geopy not found — geocoding disabled. Run: pip install geopy")
        except Exception as e:
            print(f" Geocoder setup failed ({e}) — geocoding disabled.")

    def _parse_entity(self, e: Dict, min_score: float) -> Optional[Dict]:
        """Normalise a raw pipeline entity dict.

        REMOVAL: Removed dead fallback e.get('entity') — that key does not
        exist in HuggingFace pipeline output when aggregation_strategy='simple'.
        """
        score = float(e['score'])
        if score < min_score:
            return None
        return {
            'entity_type': e.get('entity_group'),
            'entity_text': e.get('word'),
            'score':       score,
            'start':       e.get('start'),
            'end':         e.get('end'),
        }

    def _geocode_location(self, location_text: str) -> Dict:
        if not location_text or not self.geocoder:
            return {}

        cache_key = location_text.lower().strip()
        if cache_key in self._geocode_cache:
            return self._geocode_cache[cache_key]

        try:
            time.sleep(1)
            result = self.geocoder.geocode(location_text, timeout=10)
            coords = {
                'latitude':  result.latitude,
                'longitude': result.longitude,
                'address':   result.address,
            } if result else {}
        except (self._GeocoderTimedOut, self._GeocoderServiceError) as e:
            print(f"   Geocoding error for '{location_text}': {e}")
            coords = {}

        self._geocode_cache[cache_key] = coords
        return coords

    #  Public: model persistence ─

    def save_model(self, save_directory: str) -> None:
        """Save the NER model and tokenizer for later reuse.

        ADDITION: Serializes as safetensors format for faster future loading.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.ner_pipeline.model.save_pretrained(save_directory, safe_serialization=True)
        self.ner_pipeline.tokenizer.save_pretrained(save_directory)
        print(f"✓ NER model saved to {save_directory}/")

    @classmethod
    def load_model(cls, save_directory: str, **kwargs) -> 'MultilingualNER':
        """Load a previously saved NER model.

        ADDITION: Avoids re-downloading; significantly speeds up inference start.
        """
        return cls(model_name=save_directory, **kwargs)

    #  Public: NER ─

    def extract_entities(self, text: str, min_score: float = 0.5) -> List[Dict]:
        if not text or len(text.strip()) < 5:
            return []
        try:
            entities = self.ner_pipeline(text[:self.max_length])
            return [p for e in entities for p in [self._parse_entity(e, min_score)] if p]
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return []

    def extract_batch(self, texts: List[str], batch_size: int = None,
                      show_progress: bool = True,
                      max_texts: Optional[int] = None,
                      min_score: float = 0.5) -> List[List[Dict]]:
        if max_texts and len(texts) > max_texts:
            indices = np.random.choice(len(texts), max_texts, replace=False)
            texts   = [texts[i] for i in indices]

        batch_size = batch_size or self.batch_size
        results    = []
        total      = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            if show_progress:
                print(f"  NER {min(i + batch_size, total)}/{total}...", end='\r')
            try:
                truncated  = [t[:self.max_length] if t and len(t) > self.max_length else t
                              for t in batch]
                valid_mask = [bool(t and len(t.strip()) >= 5) for t in truncated]
                non_empty  = [t for t, v in zip(truncated, valid_mask) if v]

                if non_empty:
                    batch_output = self.ner_pipeline(non_empty)
                    out_iter = iter(batch_output)
                    for is_valid in valid_mask:
                        if is_valid:
                            raw = next(out_iter)
                            results.append([
                                p for e in (raw if isinstance(raw, list) else [raw])
                                for p in [self._parse_entity(e, min_score)] if p
                            ])
                        else:
                            results.append([])
                else:
                    results.extend([[]] * len(batch))

            except Exception as e:
                print(f"\n  Batch error: {e}")
                for text in batch:
                    results.append(self.extract_entities(text, min_score=min_score))

        if show_progress:
            print(f"\n✓ NER complete: {total:,} texts processed")
        return results

    #  Public: Geoparsing

    def geoparse_entities(self, entities_batch: List[List[Dict]]) -> List[List[Dict]]:
        if not self.geocoder:
            print(" Geocoder not initialised — returning entities without coordinates.")
            return entities_batch

        return [
            [
                {**e, **self._geocode_location(e['entity_text'])}
                if e.get('entity_type') == 'LOC' and e.get('entity_text')
                else e
                for e in entities
            ]
            for entities in entities_batch
        ]

    def geoparse_to_dataframe(self, df, text_column="combined_text", min_score=0.5):
        """
        Full geoparsing pipeline on a DataFrame column, processed in chunks
        to avoid GPU memory overflow on large datasets.

        Optimizations:
          - No chunk copy – works directly on slices of the original DataFrame.
          - Pre‑allocates columns with None to avoid repeated dtype inference.
          - Uses direct slice assignment for list columns (no pd.Series wrapper).
          - Reduces print frequency (uses tqdm progress bar if available).
          - Clears geocode cache only every 10 chunks to reduce overhead.
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")

        total_rows = len(df)

        # Pre‑allocate columns with None (object dtype will be inferred)
        df['all_entities'] = None
        df['loc_entities'] = None
        df['first_loc_text'] = None
        df['first_lat'] = None
        df['first_lon'] = None

        # Force object dtype for list columns to avoid later conversion issues
        for col in ['all_entities', 'loc_entities']:
            df[col] = df[col].astype(object)

        # Optional: tqdm progress bar (if installed)
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_rows, desc="Geoparsing", unit="rows", leave=False)
            use_tqdm = True
        except ImportError:
            pbar = None
            use_tqdm = False

        cache_clear_interval = 10   # clear geocode cache every N chunks
        chunk_count = 0

        for start in range(0, total_rows, self.batch_size):
            end = min(start + self.batch_size, total_rows)
            chunk_idx = slice(start, end)          # slice for direct indexing
            texts = df.iloc[chunk_idx][text_column].tolist()

            # Run NER on the chunk
            entities_batch = self.extract_batch(texts, min_score=min_score, show_progress=False)

            # Geocode if enabled
            if self.geocoder:
                entities_batch = self.geoparse_entities(entities_batch)

            for i, (abs_idx, ents) in enumerate(zip(range(start, end), entities_batch)):
              df.at[abs_idx, 'all_entities'] = ents

              locs = [e for e in ents if e.get('entity_type') == 'LOC' and 'latitude' in e]
              df.at[abs_idx, 'loc_entities'] = locs

              if locs:
                  df.at[abs_idx, 'first_loc_text'] = locs[0]['entity_text']
                  df.at[abs_idx, 'first_lat']      = locs[0].get('latitude')
                  df.at[abs_idx, 'first_lon']      = locs[0].get('longitude')

            # Periodic cache clearing (reduces memory usage)
            chunk_count += 1
            if self.geocoder and hasattr(self, '_geocode_cache') and chunk_count % cache_clear_interval == 0:
                self._geocode_cache.clear()

            # Update progress
            if use_tqdm:
                pbar.update(end - start)
            else:
                print(f"  Geoparsed rows {start+1}–{end} / {total_rows}", end="\r")

        if use_tqdm:
            pbar.close()
        else:
            print()   # newline after final print

        print(f"✓ Geoparsing complete: {total_rows:,} texts processed")
        return df

    #  Public: Analysis

    def analyze_entities(self, entities_list: List[List[Dict]]) -> Dict:
        print("\n" + "=" * 80)
        print("NAMED ENTITY ANALYSIS")
        print("=" * 80)

        all_entities = [e for entities in entities_list for e in entities]
        if not all_entities:
            print("No entities found.")
            return {}

        type_counts = Counter(e['entity_type'] for e in all_entities)
        print("\nEntity Types:")
        for etype, count in type_counts.most_common():
            pct = (count / len(all_entities)) * 100
            print(f"  {self.entity_types.get(etype, etype)}: {count:,} ({pct:.2f}%)वरुन)")

        entity_texts = Counter((e['entity_text'] or '').lower() for e in all_entities)
        print("\nMost Common Entities (Top 20):")
        for entity, count in entity_texts.most_common(20):
            print(f"  {entity}: {count:,}")

        scores = [e['score'] for e in all_entities]
        print(f"\nConfidence: mean={np.mean(scores):.3f}, "
              f"median={np.median(scores):.3f}, "
              f"min={np.min(scores):.3f}, max={np.max(scores):.3f}")

        return {
            'total_entities':    len(all_entities),
            'type_distribution': dict(type_counts),
            'top_entities':      dict(entity_texts.most_common(50)),
            'avg_confidence':    float(np.mean(scores)),
            'median_confidence': float(np.median(scores)),
        }

#
class MultilingualSentiment:
    """Sentiment analysis with probability distribution support."""

    def __init__(self, model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
                 device: int = -1, batch_size: int = 64):
        """
        device : int, optional
            -1 for CPU, 0 for first GPU, etc. If None, auto‑detect.
        batch_size : int
            Number of texts per batch (adjust based on GPU memory).
        """
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError("transformers required: pip install transformers")

        if device is None:
            import torch
            device = 0 if torch.cuda.is_available() else -1

        print(f"Loading sentiment model: {model_name} (device={'GPU' if device >= 0 else 'CPU'})...")
        self.sentiment_pipeline = hf_pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device,
            return_all_scores=True,
            batch_size=batch_size,
        )
        self.label_map = {
            'LABEL_0': 'Negative',
            'LABEL_1': 'Neutral',
            'LABEL_2': 'Positive',
        }
        print("✓ Sentiment model loaded!")

    def save_model(self, save_directory: str) -> None:
        """Save the sentiment model for reuse without re-downloading.

        ADDITION: Avoids re-downloading the model on each instantiation.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.sentiment_pipeline.model.save_pretrained(save_directory, safe_serialization=True)
        self.sentiment_pipeline.tokenizer.save_pretrained(save_directory)
        print(f"✓ Sentiment model saved to {save_directory}/")

    @classmethod
    def load_saved(cls, save_directory: str) -> 'MultilingualSentiment':
        """Load a previously saved sentiment model."""
        return cls(model_name=save_directory)

    def get_sentiment_probabilities(self, text: str) -> Dict[str, float]:
        """Return the full probability distribution over all sentiment labels.

        ADDITION: Returns raw probabilities for calibration and threshold tuning,
        not just the top label.
        """
        if not text or len(text.strip()) < 5:
            return {v: 0.0 for v in self.label_map.values()}
        try:
            all_scores = self.sentiment_pipeline(text[:512])[0]
            return {self.label_map.get(s['label'], s['label']): s['score']
                    for s in all_scores}
        except Exception:
            return {v: 0.0 for v in self.label_map.values()}

    def analyze_sentiment(self, text: str) -> Dict:
        if not text or len(text.strip()) < 5:
            return {'sentiment': 'unknown', 'score': 0.0}
        try:
            all_scores = self.sentiment_pipeline(text[:512])[0]
            best = max(all_scores, key=lambda x: x['score'])
            return {
                'sentiment': self.label_map.get(best['label'], best['label']),
                'score':     best['score'],
            }
        except Exception:
            return {'sentiment': 'unknown', 'score': 0.0}

    def analyze_batch(self, texts: List[str], batch_size: int = 16,
                      show_progress: bool = True,
                      max_texts: Optional[int] = None) -> pd.DataFrame:
        """Analyze sentiment for a batch of texts.

        ADDITION: Every input text now receives an output row, including
        short texts that receive {'sentiment': 'unknown', 'score': 0.0},
        preventing silent index misalignment.
        """
        if max_texts and len(texts) > max_texts:
            indices = np.random.choice(len(texts), max_texts, replace=False)
            texts   = [texts[i] for i in indices]

        results = []
        total   = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            if show_progress:
                print(f"  Sentiment {min(i + batch_size, total)}/{total}...", end='\r')
            for text in batch:
                # always append one result per input text
                results.append(self.analyze_sentiment(text))

        if show_progress:
            print(f"\n✓ Sentiment complete: {total:,} texts")
        return pd.DataFrame(results)

    def analyze_sentiments(self, df: pd.DataFrame) -> Dict:
        print("\n" + "=" * 80)
        print("SENTIMENT ANALYSIS")
        print("=" * 80)

        sentiment_counts = df['sentiment'].value_counts()
        total = len(df)

        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            pct = (count / total) * 100
            print(f"  {sentiment}: {count:,} ({pct:.2f}%)")

        print("\nAverage Confidence by Sentiment:")
        for sentiment in sentiment_counts.index:
            avg    = df[df['sentiment'] == sentiment]['score'].mean()
            median = df[df['sentiment'] == sentiment]['score'].median()
            print(f"  {sentiment}: mean={avg:.3f}, median={median:.3f}")

        return {
            'distribution':              sentiment_counts.to_dict(),
            'avg_score_by_sentiment':    df.groupby('sentiment')['score'].mean().to_dict(),
            'median_score_by_sentiment': df.groupby('sentiment')['score'].median().to_dict(),
        }


class MultilingualTopicModeling:
    """BERTopic with reproducible UMAP, outlier reduction, and visualization."""

    def __init__(self, language: str = 'multilingual',
                 min_topic_size: int = 10,
                 nr_topics: Optional[int] = None,
                 min_cluster_size: Optional[int] = None):
        self.language         = language
        self.min_topic_size   = min_topic_size
        # prefer min_cluster_size for controlling topic granularity
        self.min_cluster_size = min_cluster_size
        # nr_topics is deprecated; use min_cluster_size instead
        if nr_topics is not None:
            print("  DeprecationWarning: nr_topics merges topics post-hoc. "
                  "Prefer min_cluster_size to control granularity at the cluster level.")
        self.nr_topics  = nr_topics
        self.topic_model = None
        self.label_map   = {}
        self.words_map   = {}
        self._load_model()

    def _load_model(self):
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            from sklearn.feature_extraction.text import CountVectorizer
            from umap import UMAP
            from hdbscan import HDBSCAN
        except ImportError:
            raise ImportError(
                "bertopic, sentence-transformers, umap-learn, and hdbscan required.")

        print(f"Initializing BERTopic ({self.language})...")

        if self.language == 'multilingual':
            embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        elif self.language == 'dutch':
            embedding_model = SentenceTransformer('GroNLP/bert-base-dutch-cased')
        else:
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # explicit random_state on UMAP for reproducibility
        umap_model = UMAP(n_neighbors=15, n_components=5,
                          min_dist=0.0, metric='cosine',
                          random_state=42)

        # optional min_cluster_size for granularity control
        hdbscan_kwargs = {'min_cluster_size': self.min_cluster_size} \
            if self.min_cluster_size else {}
        hdbscan_model = HDBSCAN(min_samples=10, prediction_data=True, **hdbscan_kwargs)

        # Dutch stop words added to CountVectorizer
        dutch_stop_words = list(TextPreprocessor.DUTCH_STOPWORDS)
        vectorizer_model = CountVectorizer(
            stop_words=dutch_stop_words, ngram_range=(1, 2), min_df=2
        )

        self.topic_model = BERTopic(
            embedding_model      = embedding_model,
            umap_model           = umap_model,
            hdbscan_model        = hdbscan_model,
            vectorizer_model     = vectorizer_model,
            min_topic_size       = self.min_topic_size,
            nr_topics            = self.nr_topics,
            verbose              = False,
            calculate_probabilities = False,
        )
        print("✓ BERTopic initialized!")

    def transform(self, texts: List[str]) -> List[int]:
        """
        Assign topics to new texts using the fitted topic model.
        The model must have been fitted first via fit_transform().
        """
        if self.topic_model is None:
            raise RuntimeError("Model not fitted. Call fit_transform() first.")
        # BERTopic.transform returns (topics, probabilities)
        topics, _ = self.topic_model.transform(texts)
        return topics

    def fit_transform(self, texts: List[str],
                      sample_size: Optional[int] = None) -> Tuple[List[int], Optional[np.ndarray]]:
        valid_texts = [t for t in texts if t and len(t.strip()) > 10]
        if len(valid_texts) < 10:
            print("Not enough valid texts for topic modeling")
            return [], None

        if sample_size and len(valid_texts) > sample_size:
            indices     = np.random.choice(len(valid_texts), sample_size, replace=False)
            valid_texts = [valid_texts[i] for i in indices]

        print(f"Fitting topic model on {len(valid_texts):,} texts...")
        topics, probs = self.topic_model.fit_transform(valid_texts)

        n_topics = len(set(topics)) - (1 if -1 in topics else 0)
        # removed the duplicate print statement
        print(f"✓ Found {n_topics} topics")

        self.label_map = {}
        self.words_map = {}
        topic_info = self.topic_model.get_topic_info()
        for _, row in topic_info.iterrows():
            tid = int(row['Topic'])
            if tid == -1:
                continue
            words = self.topic_model.get_topic(tid) or []
            self.words_map[tid] = words
            self.label_map[tid] = ', '.join(w for w, _ in words[:3]) or f'Topic {tid}'
        print(f"✓ Label map built for {len(self.label_map)} topics")

        return topics, probs

    def reduce_outliers(self, documents: List[str], topics: List[int],
                        strategy: str = 'c-tf-idf') -> List[int]:
        """Reassign outlier documents (topic -1) to nearest topics.

        ADDITION: Addresses the common problem where outliers exceed 74% of
        the dataset with default HDBSCAN settings. BERTopic's reduce_outliers
        provides multiple strategies: 'c-tf-idf', 'embeddings', 'distributions'.
        """
        if self.topic_model is None:
            raise RuntimeError("Call fit_transform() before reduce_outliers().")
        updated_topics = self.topic_model.reduce_outliers(documents, topics, strategy=strategy)
        self.topic_model.update_topics(documents, topics=updated_topics)
        n_remaining = sum(1 for t in updated_topics if t == -1)
        print(f"✓ Outliers reduced. Remaining outliers: {n_remaining:,}")
        return updated_topics

    def get_topic_info(self) -> pd.DataFrame:
        return self.topic_model.get_topic_info()

    def get_topic_words(self, topic_id: int, top_n: int = 10) -> List[Tuple[str, float]]:
        try:
            return self.topic_model.get_topic(topic_id)[:top_n]
        except Exception:
            return []

    def get_representative_docs(self, topic_id: int) -> List[str]:
        """Return representative documents for a given topic.

        ADDITION: Essential for auditing and validating topic quality.
        """
        if self.topic_model is None:
            raise RuntimeError("Call fit_transform() first.")
        try:
            return self.topic_model.get_representative_docs(topic_id)
        except Exception as e:
            print(f"  Could not retrieve representative docs: {e}")
            return []

    def visualize_topics(self, output_dir: str = 'topic_output'):
        os.makedirs(output_dir, exist_ok=True)
        try:
            fig = self.topic_model.visualize_topics()
            fig.write_html(f'{output_dir}/topics_visualization.html')
            print(f"✓ Topics visualization saved to {output_dir}/topics_visualization.html")
        except Exception as e:
            print(f"Error: {e}")

    def visualize_barchart(self, top_n_topics: int = 10, output_dir: str = 'topic_output'):
        os.makedirs(output_dir, exist_ok=True)
        try:
            fig = self.topic_model.visualize_barchart(top_n_topics=top_n_topics)
            fig.write_html(f'{output_dir}/topics_barchart.html')
            print(f"✓ Topics barchart saved to {output_dir}/topics_barchart.html")
        except Exception as e:
            print(f"Error: {e}")

    def visualize_heatmap(self, output_dir: str = 'topic_output'):
        os.makedirs(output_dir, exist_ok=True)
        try:
            fig = self.topic_model.visualize_heatmap()
            fig.write_html(f'{output_dir}/topics_heatmap.html')
            print(f"✓ Topics heatmap saved to {output_dir}/topics_heatmap.html")
        except Exception as e:
            print(f"Error: {e}")

    def visualize_hierarchy(self, output_dir: str = 'topic_output'):
        os.makedirs(output_dir, exist_ok=True)
        try:
            fig = self.topic_model.visualize_hierarchy()
            fig.write_html(f'{output_dir}/topics_hierarchy.html')
            print(f"✓ Hierarchy saved to {output_dir}/topics_hierarchy.html")
        except Exception as e:
            print(f"Error: {e}")

    def analyze_topics(self, topics: List[int], texts: List[str]) -> Dict:
        print("\n" + "=" * 80)
        print("TOPIC MODELING ANALYSIS")
        print("=" * 80)

        topic_info = self.get_topic_info()
        num_topics = len(topic_info[topic_info['Topic'] != -1])
        outliers   = len([t for t in topics if t == -1])

        print(f"\nTotal Topics Found: {num_topics}")
        print(f"Outliers (Topic -1): {outliers} ({outliers/len(topics)*100:.2f}%)")

        print("\nTop 10 Topics:")
        for idx, row in topic_info.head(11).iterrows():
            if row['Topic'] == -1:
                continue
            print(f"\n  Topic {row['Topic']}: {row['Name']}")
            print(f"    Count: {row['Count']} ({row['Count']/len(topics)*100:.2f}%)")
            top_words = self.get_topic_words(row['Topic'], top_n=5)
            if top_words:
                words_str = ", ".join([f"{w} ({s:.3f})" for w, s in top_words])
                print(f"    Top words: {words_str}")

        return {
            'num_topics':         num_topics,
            'num_outliers':       outliers,
            'outlier_percentage': outliers / len(topics) * 100,
            'topic_info':         topic_info.to_dict('records'),
        }

# SECTION 5: QUERY / FILTER CLASSES

from typing import Optional
import pandas as pd


class AlertsFilter:
    """Filter alerts based on various criteria."""

    def filter_alerts(
        self,
        df: pd.DataFrame,
        cause        = None,
        effect       = None,
        route        = None,
        min_duration = None,
        date_range   = None,
    ) -> pd.DataFrame:

        filtered_df = df.copy()

        #  Cause ─
        if cause:
            filtered_df = filtered_df[filtered_df['cause'] == cause]
            print(f"Filtered by cause       : {cause}")

        #  Effect
        if effect:
            filtered_df = filtered_df[filtered_df['effect'] == effect]
            print(f"Filtered by effect      : {effect}")

        #  Route ─
        if route:
            filtered_df = filtered_df[filtered_df['route'] == str(route)]
            print(f"Filtered by route       : {route}")

        #  Minimum duration
        if min_duration is not None and 'alert_duration_min' in filtered_df.columns:
            col_dtype = filtered_df['alert_duration_min'].dtype

            if pd.api.types.is_timedelta64_dtype(col_dtype):
                # Column is timedelta64 — convert int minutes to Timedelta
                threshold = pd.Timedelta(minutes=min_duration)

            elif pd.api.types.is_numeric_dtype(col_dtype):
                # Column already stores numeric minutes — compare directly
                threshold = min_duration

            else:
                raise TypeError(
                    f"alert_duration_min has unexpected dtype '{col_dtype}'. "
                    f"Expected timedelta64[ns] or numeric. "
                    f"Convert the column before filtering."
                )

            filtered_df = filtered_df[filtered_df['alert_duration_min'] >= threshold]
            print(f"Filtered by min duration: {min_duration} minutes  "
                  f"(threshold={threshold})")

        elif min_duration is not None:
            print(f"  ⚠ min_duration specified but 'alert_duration_min' column not found — skipped")

        #  Date range
        if date_range is not None and 'id_time' in filtered_df.columns:
            start_date, end_date = date_range

            # Ensure id_time is datetime — coerce if needed
            if not pd.api.types.is_datetime64_any_dtype(filtered_df['id_time']):
                filtered_df['id_time'] = pd.to_datetime(
                    filtered_df['id_time'], errors='coerce'
                )

            start_ts = pd.Timestamp(start_date)
            end_ts   = pd.Timestamp(end_date)

            filtered_df = filtered_df[
                (filtered_df['id_time'] >= start_ts) &
                (filtered_df['id_time'] <= end_ts)
            ]
            print(f"Filtered by date range  : {start_date} → {end_date}")

        elif date_range is not None:
            print(f"  ⚠ date_range specified but 'id_time' column not found — skipped")

        print(f"\nTotal alerts after filtering: {len(filtered_df):,}")
        return filtered_df

    def filter_by_language(
        self, df: pd.DataFrame, language_code: str
    ) -> pd.DataFrame:
        """Filter alerts by detected language code."""
        col = 'language_code' if 'language_code' in df.columns else \
              'language'      if 'language'      in df.columns else None

        if col is None:
            raise ValueError(
                "No language column found. "
                "Expected 'language_code' or 'language'. "
                "Run language detection first."
            )

        result = df[df[col] == language_code].copy()
        print(f"Filtered by language '{language_code}': {len(result):,} alerts")
        return result

    def filter_by_active_now(
        self,
        df             : pd.DataFrame,
        reference_time : Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Return only alerts whose active period covers a reference timestamp."""
        if reference_time is None:
            reference_time = pd.Timestamp.now()

        required = ['active_period_start', 'active_period_end']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Columns required but not found: {missing}"
            )

        start = pd.to_datetime(df['active_period_start'], errors='coerce')
        end   = pd.to_datetime(df['active_period_end'],   errors='coerce')

        mask   = (start <= reference_time) & (end >= reference_time)
        result = df[mask].copy()
        print(f"Active at {reference_time}: {len(result):,} alerts")
        return result

    def filter_by_agency_id(
        self, df: pd.DataFrame, agency_id: str
    ) -> pd.DataFrame:
        """Filter alerts by agency_id."""
        if 'agency_id' not in df.columns:
            raise ValueError("No 'agency_id' column found.")

        result = df[df['agency_id'] == agency_id].copy()
        print(f"Filtered by agency '{agency_id}': {len(result):,} alerts")
        return result

class NLPQueryEngine:
    """Unified interface for single alert or batch NLP analysis."""

    def __init__(self):
        self.lang_detector = LanguageDetector()
        self.ner           = MultilingualNER()
        self.sentiment     = MultilingualSentiment()
        self.topic_modeler = None
        self._memo: Dict[str, Dict] = {}

    def analyze_alert(self, text: str) -> Dict:
        """Analyze a single alert text.

        ADDITION: Memoized on text hash to avoid rerunning expensive NER+sentiment
        on duplicate alerts, which are extremely common in GTFS-RT snapshots.
        REMOVAL: Removed the empty `pass` placeholder topic block.
        """
        cache_key = hashlib.sha1(text.encode('utf-8', errors='ignore')).hexdigest()
        if cache_key in self._memo:
            return self._memo[cache_key]

        result = {}
        lang_code, confidence = self.lang_detector.detect_language(text)
        result['language'] = {
            'code':       lang_code,
            'name':       self.lang_detector.lang_names.get(lang_code, lang_code),
            'confidence': confidence,
        }
        result['sentiment'] = self.sentiment.analyze_sentiment(text)
        result['entities']  = self.ner.extract_entities(text)

        self._memo[cache_key] = result
        return result

    def explain_alert(self, text: str) -> str:
        """Return a human-readable inline summary of entities found in the text.

        ADDITION: Provides entity highlights for dashboard integration.
        """
        result = self.analyze_alert(text)
        entities = result.get('entities', [])
        if not entities:
            return f"[No entities found] | Sentiment: {result['sentiment']['sentiment']}"
        entity_summary = ', '.join(
            f"{e['entity_type']}:{e['entity_text']}" for e in entities[:5]
        )
        return (f"Entities: {entity_summary} | "
                f"Sentiment: {result['sentiment']['sentiment']} "
                f"({result['sentiment']['score']:.2f}) | "
                f"Language: {result['language']['name']}")

    def analyze_batch(self, texts: List[str], fit_topic_model: bool = False) -> pd.DataFrame:
        lang_df     = self.lang_detector.detect_batch(texts, show_progress=True)
        sent_df     = self.sentiment.analyze_batch(texts, show_progress=True)
        ner_results = self.ner.extract_batch(texts, show_progress=True)

        df = pd.concat([lang_df, sent_df], axis=1)
        df['entities'] = ner_results

        if fit_topic_model:
            self.topic_modeler = MultilingualTopicModeling(min_topic_size=5)
            topics, _          = self.topic_modeler.fit_transform(texts)
            df['topic']        = topics
            topic_info         = self.topic_modeler.get_topic_info()
            topic_map          = dict(zip(topic_info['Topic'], topic_info['Name']))
            df['topic_name']   = df['topic'].map(topic_map).fillna('Outlier')

        return df

# SECTION 6: ANALYSIS CLASSES

class GTFSAlertExplorer:
    """Comprehensive data exploration for GTFS-RT alerts."""

    def __init__(self, alerts_df: pd.DataFrame):
        self.df = alerts_df.copy()
        self.exploration_results = {}
        self._cached_stats = None   # None means "not computed yet"
        for col in ['feed_timestamp', 'active_period_start', 'active_period_end', 'id_time']:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

    def explore_basic_statistics(self, force_refresh: bool = False):
        # fixed the `if self._cached_stats` bug — empty dict {} is falsy,
        # so the old check would recompute even when a valid (empty) result was cached.
        if self._cached_stats is not None and not force_refresh:
            self._print_basic_stats(self._cached_stats)
            return self._cached_stats

        print("=" * 80)
        print("BASIC STATISTICS")
        print("=" * 80)

        stats = {
            'total_alerts':     len(self.df),
            'unique_alert_ids': self.df['alert_id'].nunique() if 'alert_id' in self.df.columns else 0,
            'date_range':       None,
            'missing_data':     {},
        }

        if 'feed_timestamp' in self.df.columns:
            valid = self.df['feed_timestamp'].dropna()
            if not valid.empty:
                stats['date_range'] = {'start': valid.min(), 'end': valid.max()}

        missing_info = self.df.isnull().sum()
        total_rows   = len(self.df)
        for col, missing_count in missing_info[missing_info > 0].items():
            stats['missing_data'][col] = {
                'count':      int(missing_count),
                'percentage': f"{(missing_count / total_rows) * 100:.2f}%",
            }

        self._cached_stats = stats
        self._print_basic_stats(stats)
        self.exploration_results['basic_stats'] = stats
        return stats

    def _print_basic_stats(self, stats: Dict):
        print(f"\nTotal Alerts:      {stats['total_alerts']:,}")
        print(f"Unique Alert IDs:  {stats['unique_alert_ids']:,}")
        if stats['date_range']:
            print(f"Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        if stats['missing_data']:
            print("\nMissing Data (Top 10):")
            for col, info in sorted(
                stats['missing_data'].items(),
                key=lambda x: x[1]['count'], reverse=True
            )[:10]:
                print(f"  - {col}: {info['count']} ({info['percentage']})")

    def explore_text_content(self):
        print("\n" + "=" * 80)
        print("TEXT CONTENT ANALYSIS")
        print("=" * 80)
        text_stats = {}

        if 'description_text' in self.df.columns:
            desc     = self.df['description_text'].fillna('')
            nonempty = desc != ''
            text_stats['description'] = {
                'total_non_empty': nonempty.sum(),
                'avg_length':      desc.str.len().mean(),
                'max_length':      desc.str.len().max(),
                'min_length':      desc[nonempty].str.len().min() if nonempty.any() else 0,
                'median_length':   desc.str.len().median(),
                'std_length':      desc.str.len().std(),
            }
            d = text_stats['description']
            print(f"\nDescription Text:")
            print(f"  Non-empty alerts: {d['total_non_empty']:,}")
            print(f"  Avg length:       {d['avg_length']:.2f} chars")
            print(f"  Median length:    {d['median_length']:.2f} chars")
            print(f"  Max length:       {d['max_length']}")

        if 'header_text' in self.df.columns:
            hdr      = self.df['header_text'].fillna('')
            nonempty = hdr != ''
            text_stats['header'] = {
                'total_non_empty': nonempty.sum(),
                'avg_length':      hdr.str.len().mean(),
            }
            print(f"\nHeader Text:")
            print(f"  Non-empty alerts: {text_stats['header']['total_non_empty']:,}")
            print(f"  Avg length:       {text_stats['header']['avg_length']:.2f} chars")

        self.exploration_results['text_stats'] = text_stats
        return text_stats

    def explore_alert_causes_effects(self):
        print("\n" + "=" * 80)
        print("CAUSES AND EFFECTS ANALYSIS")
        print("=" * 80)
        total = len(self.df)

        if 'cause' in self.df.columns:
            print("\nTop Alert Causes:")
            cause_counts = self.df['cause'].value_counts()
            for cause, count in cause_counts.head(10).items():
                print(f"  {cause}: {count:,} ({count/total*100:.2f}%)")
            self.exploration_results['causes'] = cause_counts.to_dict()

        if 'effect' in self.df.columns:
            print("\nTop Alert Effects:")
            effect_counts = self.df['effect'].value_counts()
            for effect, count in effect_counts.head(10).items():
                print(f"  {effect}: {count:,} ({count/total*100:.2f}%)")
            self.exploration_results['effects'] = effect_counts.to_dict()

        return self.exploration_results

    def explore_temporal_patterns(self):
        print("\n" + "=" * 80)
        print("TEMPORAL PATTERNS")
        print("=" * 80)

        if 'feed_timestamp' not in self.df.columns:
            print("No timestamp data available")
            return

        valid = self.df['feed_timestamp'].dropna()
        if valid.empty:
            print("No valid timestamps found")
            return

        self.df.loc[valid.index, 'hour']     = valid.dt.hour
        self.df.loc[valid.index, 'day_name'] = valid.dt.day_name()  # str label only
        self.df.loc[valid.index, 'date']     = valid.dt.date

        print("\nAlerts by Hour of Day (Top 10):")
        hourly = self.df['hour'].value_counts().sort_index().head(10)
        for hour, count in hourly.items():
            print(f"  {int(hour):02d}:00 - {count:,} alerts")

        print("\nAlerts by Day of Week:")
        day_order    = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_col      = 'day_name' if 'day_name' in self.df.columns else None
        if day_col:
            daily_sorted = self.df[day_col].value_counts().reindex(day_order, fill_value=0)
            for day, count in daily_sorted.items():
                print(f"  {day}: {count:,} alerts")
        else:
            daily_sorted = pd.Series(dtype=int)

        self.exploration_results['temporal'] = {
            'hourly': hourly.to_dict(),
            'daily':  daily_sorted.to_dict() if not daily_sorted.empty else {},
        }
        return self.exploration_results

    def explore_informed_entities(self):
        """Analyse which routes, stops, and agencies are most affected.

        ADDITION: Previously absent — this is the most operationally
        important exploratory view for a transit analyst.
        """
        print("\n" + "=" * 80)
        print("INFORMED ENTITIES ANALYSIS")
        print("=" * 80)

        results = {}

        if 'route_id' in self.df.columns:
            top_routes = self.df['route_id'].dropna().value_counts().head(15)
            print("\nTop 15 Affected Route IDs:")
            for route, count in top_routes.items():
                print(f"  {route}: {count:,}")
            results['top_routes'] = top_routes.to_dict()

        if 'stop_id' in self.df.columns:
            top_stops = self.df['stop_id'].dropna().value_counts().head(15)
            print("\nTop 15 Affected Stop IDs:")
            for stop, count in top_stops.items():
                print(f"  {stop}: {count:,}")
            results['top_stops'] = top_stops.to_dict()

        if 'agency_id' in self.df.columns:
            agency_counts = self.df['agency_id'].dropna().value_counts()
            print("\nAlerts by Agency:")
            for agency, count in agency_counts.items():
                print(f"  {agency}: {count:,}")
            results['agency_distribution'] = agency_counts.to_dict()

        self.exploration_results['informed_entities'] = results
        return results

    def by_cause(self, cause: str) -> pd.DataFrame:
        return self.df[self.df['cause'] == cause]

    def by_route(self, route_num) -> pd.DataFrame:
        return self.df[self.df['route'] == str(route_num)]

    def top_words(self, n: int = 20) -> List[Tuple[str, int]]:
        text = ' '.join(self.df['clean_text'].dropna())
        return Counter(text.split()).most_common(n)

    def summary(self):
        print("=" * 50)
        print("GTFS-RT Alerts Summary")
        print("=" * 50)
        print(f"Total records: {len(self.df):,}")
        if 'cause' in self.df.columns:
            print("\nTop 5 Causes:")
            print(self.df['cause'].value_counts().head())

    def run_full_exploration(self, output_dir: str = 'exploration_output'):
        print("\n" + "=" * 80)
        print("RUNNING COMPLETE DATA EXPLORATION")
        print("=" * 80)
        self.explore_basic_statistics()
        self.explore_text_content()
        self.explore_alert_causes_effects()
        self.explore_temporal_patterns()
        self.explore_informed_entities()   # now included in full run
        return self.exploration_results


class AlertsAnalyzer:
    """Statistical analysis of GTFS-RT alerts."""

    def analyze_basic(self, df):
        if df is None or df.empty:
            return df

        print("=" * 60)
        print("BASIC ALERTS ANALYSIS")
        print("=" * 60)
        print(f"\n1. BASIC STATISTICS:")
        print(f"   Total records: {len(df):,}")

        if 'feed_timestamp' in df.columns:
            print(f"   Date range: {df['feed_timestamp'].min()} to {df['feed_timestamp'].max()}")

        print("\n2. MISSING VALUES:")
        missing     = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df  = pd.DataFrame({'Missing': missing, 'Percentage': missing_pct})
        print(missing_df[missing_df['Missing'] > 0])

        if 'cause' in df.columns:
            print("\n3. Alert Causes Distribution:")
            print(df['cause'].value_counts())

        if 'effect' in df.columns:
            print("\n4. Alert Effects Distribution:")
            print(df['effect'].value_counts())

        if 'text_length' in df.columns:
            print("\n5. Text Statistics:")
            print(f"   Avg text length:  {df['text_length'].mean():.2f} characters")
            print(f"   Avg word count:   {df['word_count'].mean():.2f} words")
            print(f"   Median text len:  {df['text_length'].median():.2f} characters")

        if 'alert_duration_min' in df.columns:
            valid = df['alert_duration_min'].dropna()
            if len(valid) > 0:
                print("\n6. Duration Statistics:")
                print(f"   Avg duration:    {valid.mean():.2f} minutes")
                print(f"   Median duration: {valid.median():.2f} minutes")
                print(f"   Max duration:    {valid.max():.2f} minutes")

        if 'route' in df.columns:
            print("\n7. Top 10 Routes with Alerts:")
            print(df['route'].value_counts().head(10))

        return df

    def analyze_temporal_patterns(self, df):
        if df is None or df.empty or 'id_time' not in df.columns:
            return df

        df['id_time'] = pd.to_datetime(df['id_time'], errors='coerce')
        valid = df['id_time'].dropna()
        if valid.empty:
            return df

        df.loc[valid.index, 'hour']     = valid.dt.hour
        df.loc[valid.index, 'day_name'] = valid.dt.day_name()  # str label only
        df.loc[valid.index, 'date']     = valid.dt.date

        print("\n" + "=" * 60)
        print("TEMPORAL PATTERN ANALYSIS")
        print("=" * 60)

        print("\n1. Alerts by Hour of Day:")
        print(df['hour'].value_counts().sort_index())

        print("\n2. Alerts by Day of Week:")
        print(df['day_name'].value_counts() if 'day_name' in df.columns else "n/a")

        print("\n3. Daily Alert Trend (last 10 days):")
        print(df.groupby('date').size().tail(10))

        return df

    def analyze_route_alerts(self, df, route_number=None):
        if route_number:
            route_df = df[df['route'] == str(route_number)]
            print(f"\nAnalysis for Route {route_number}: {len(route_df):,} alerts")
        else:
            route_df = df[df['route'].notna()]
            print(f"\nAll routes with identified numbers: {len(route_df):,} alerts")

        if route_df.empty:
            print("No alerts found")
            return route_df

        print("\nCause distribution:")
        print(route_df['cause'].value_counts())
        print("\nEffect distribution:")
        print(route_df['effect'].value_counts())

        if 'alert_duration_min' in route_df.columns:
            valid = route_df['alert_duration_min'].dropna()
            if len(valid) > 0:
                print(f"\nAvg alert duration: {valid.mean():.2f} minutes")

        return route_df

    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series,
                              cv: int = 5) -> Dict:
        """Evaluate a fitted model with cross-validation metrics.

        ADDITION: The pipeline trained models but had no evaluation step.
        Returns F1, AUC, precision, recall for classification models.
        """
        from sklearn.model_selection import cross_validate as sk_cv
        from sklearn.metrics import make_scorer, f1_score, roc_auc_score, \
            precision_score, recall_score

        scoring = {
            'f1':        make_scorer(f1_score, average='weighted', zero_division=0),
            'precision': make_scorer(precision_score, average='weighted', zero_division=0),
            'recall':    make_scorer(recall_score,    average='weighted', zero_division=0),
        }

        results = sk_cv(model, X, y, cv=cv, scoring=scoring, return_train_score=False)
        summary = {metric: {'mean': float(scores.mean()), 'std': float(scores.std())}
                   for metric, scores in results.items() if not metric.startswith('fit_')}

        print("\n" + "=" * 60)
        print("CROSS-VALIDATION RESULTS")
        print("=" * 60)
        for metric, vals in summary.items():
            print(f"  {metric}: {vals['mean']:.4f} ± {vals['std']:.4f}")
        return summary


# SECTION 7: VISUALIZATION

class VisualizationConfig:
    STYLE         = 'seaborn-v0_8-whitegrid'
    FIGURE_DPI    = 300
    SAVE_FORMAT   = 'png'
    COLORS = {
        'primary':   '#2E86AB',
        'secondary': '#A23B72',
        'accent':    '#F18F01',
        'success':   '#06A77D',
        'warning':   '#F77F00',
        'danger':    '#D62828',
        'neutral':   '#6C757D',
    }
    FONT_TITLE    = {'size': 16, 'weight': 'bold'}
    FONT_SUBTITLE = {'size': 14, 'weight': 'bold'}
    FONT_LABEL    = {'size': 12}


def _to_float_minutes(series: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds() / 60.0
    return pd.to_numeric(series, errors='coerce')


def _no_data(ax, msg: str = 'No data available'):
    ax.text(0.5, 0.5, msg, ha='center', va='center',
            transform=ax.transAxes, fontsize=11,
            color='#999999', style='italic')
    ax.set_xticks([])
    ax.set_yticks([])


class GTFSVisualizer:
    """Complete visualization suite for GTFS-RT alerts."""

    DAY_ORDER  = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                  'Friday', 'Saturday', 'Sunday']
    SHORT_DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    def __init__(self, df: pd.DataFrame, output_dir: str = 'visualizations',
                 interactive_mode: bool = True):
        """
        ADDITION: interactive_mode flag controls whether plt.show() is called.
        Set to False in non-interactive (CI/server) environments to prevent
        blank saved figures.
        """
        self.df              = df.copy()
        self.output_dir      = output_dir
        self.config          = VisualizationConfig()
        self.interactive_mode = interactive_mode
        os.makedirs(output_dir, exist_ok=True)
        try:
            plt.style.use(self.config.STYLE)
        except Exception:
            plt.style.use('default')
        plt.rcParams.update({
            'font.size': 10, 'axes.titlesize': 13, 'axes.labelsize': 11,
            'xtick.labelsize': 9, 'ytick.labelsize': 9,
            'legend.fontsize': 9, 'figure.titlesize': 15,
        })
        print(f"✓ GTFSVisualizer ready — {len(self.df):,} records → {self.output_dir}/")

    def _show_and_save(self, filename: str):
        """Render (optionally) and save the current figure."""
        plt.tight_layout()
        if self.interactive_mode:
            plt.show()
        path = f'{self.output_dir}/{filename}.{self.config.SAVE_FORMAT}'
        plt.savefig(path, dpi=self.config.FIGURE_DPI, bbox_inches='tight')
        plt.close('all')
        print(f"  ✓ Saved → {path}")

    #  1. Text Feature Analysis
    def plot_text_features(self):
        print("\n Plotting text features...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Text Features Analysis', **self.config.FONT_TITLE)

        if 'text_length' in self.df.columns:
            lengths = self.df['text_length'].dropna()
            if not lengths.empty:
                axes[0, 0].hist(lengths, bins=50,
                                color=self.config.COLORS['primary'],
                                edgecolor='black', alpha=0.7)
                axes[0, 0].axvline(lengths.mean(),   color='red',   linestyle='--',
                                   linewidth=2, label=f'Mean: {lengths.mean():.0f}')
                axes[0, 0].axvline(lengths.median(), color='green', linestyle='--',
                                   linewidth=2, label=f'Median: {lengths.median():.0f}')
                axes[0, 0].legend()
            else:
                _no_data(axes[0, 0])
        else:
            _no_data(axes[0, 0], 'text_length column missing')
        axes[0, 0].set_title('Text Length Distribution')
        axes[0, 0].set_xlabel('Characters')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)

        if 'word_count' in self.df.columns:
            wc = self.df['word_count'].dropna()
            if not wc.empty:
                axes[0, 1].hist(wc, bins=40,
                                color=self.config.COLORS['secondary'],
                                edgecolor='black', alpha=0.7)
                axes[0, 1].axvline(wc.mean(), color='red', linestyle='--',
                                   linewidth=2, label=f'Mean: {wc.mean():.1f}')
                axes[0, 1].legend()
            else:
                _no_data(axes[0, 1])
        else:
            _no_data(axes[0, 1], 'word_count column missing')
        axes[0, 1].set_title('Word Count Distribution')
        axes[0, 1].set_xlabel('Words')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)

        if 'text_length' in self.df.columns and 'cause' in self.df.columns:
            top_causes = self.df['cause'].value_counts().head(5).index
            groups     = [(c, self.df[self.df['cause'] == c]['text_length'].dropna())
                          for c in top_causes]
            groups     = [(c, g) for c, g in groups if not g.empty]
            if groups:
                labels_, data_ = zip(*groups)
                bp = axes[0, 2].boxplot(data_, tick_labels=labels_, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor(self.config.COLORS['accent'])
                axes[0, 2].tick_params(axis='x', rotation=30)
            else:
                _no_data(axes[0, 2], 'No cause/length data')
        else:
            _no_data(axes[0, 2], 'cause or text_length missing')
        axes[0, 2].set_title('Text Length by Top 5 Causes')
        axes[0, 2].set_xlabel('Cause')
        axes[0, 2].set_ylabel('Text Length')
        axes[0, 2].grid(True, alpha=0.3)

        if 'text_length' in self.df.columns and 'word_count' in self.df.columns:
            sub    = self.df[['text_length', 'word_count']].dropna()
            sample = sub.sample(min(1000, len(sub)), random_state=42) if len(sub) else sub
            if not sample.empty:
                axes[1, 0].scatter(sample['word_count'], sample['text_length'],
                                   alpha=0.4, s=15, color=self.config.COLORS['primary'])
                try:
                    z  = np.polyfit(sample['word_count'], sample['text_length'], 1)
                    p  = np.poly1d(z)
                    xs = np.sort(sample['word_count'].values)
                    axes[1, 0].plot(xs, p(xs), 'r--', alpha=0.8, linewidth=2,
                                    label=f'y={z[0]:.1f}x+{z[1]:.1f}')
                    axes[1, 0].legend()
                except (np.linalg.LinAlgError, ValueError):
                    pass
            else:
                _no_data(axes[1, 0])
        else:
            _no_data(axes[1, 0], 'Columns missing')
        axes[1, 0].set_title('Word Count vs Text Length')
        axes[1, 0].set_xlabel('Word Count')
        axes[1, 0].set_ylabel('Text Length')
        axes[1, 0].grid(True, alpha=0.3)

        if 'has_text' in self.df.columns:
            tc       = (self.df['has_text']
                        .value_counts()
                        .reindex([True, False], fill_value=0))
            non_zero = tc[tc > 0]
            if not non_zero.empty:
                lbl  = ['Has Text' if idx else 'No Text' for idx in non_zero.index]
                clrs = [self.config.COLORS['success'] if idx
                        else self.config.COLORS['danger'] for idx in non_zero.index]
                axes[1, 1].pie(non_zero.values, labels=lbl,
                               autopct='%1.1f%%', colors=clrs, startangle=90)
            else:
                _no_data(axes[1, 1])
        else:
            _no_data(axes[1, 1], 'has_text column missing')
        axes[1, 1].set_title('Text Availability')

        lang_col = ('language_name' if 'language_name' in self.df.columns else
                    'language'      if 'language'      in self.df.columns else None)
        if lang_col and 'text_length' in self.df.columns:
            lt = (self.df.groupby(lang_col)['text_length']
                  .mean().sort_values(ascending=False).head(10))
            if not lt.empty:
                axes[1, 2].barh(range(len(lt)), lt.values,
                                color=self.config.COLORS['secondary'])
                axes[1, 2].set_yticks(range(len(lt)))
                axes[1, 2].set_yticklabels(lt.index)
                axes[1, 2].set_xlabel('Average Characters')
                axes[1, 2].grid(True, alpha=0.3, axis='x')
            else:
                _no_data(axes[1, 2])
        else:
            _no_data(axes[1, 2], 'language_name or text_length missing')
        axes[1, 2].set_title('Avg Text Length by Language (Top 10)')

        self._show_and_save('text_features')

    #  2. Temporal features
    def plot_temporal_features(self):
        print("\n Plotting temporal features...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Temporal Features Analysis', **self.config.FONT_TITLE)

        if 'hour' in self.df.columns:
            hourly = self.df['hour'].dropna().value_counts().sort_index()
            if not hourly.empty:
                axes[0, 0].plot(hourly.index, hourly.values, marker='o',
                                linewidth=2, markersize=7,
                                color=self.config.COLORS['primary'])
                axes[0, 0].fill_between(hourly.index, hourly.values, alpha=0.25,
                                        color=self.config.COLORS['primary'])
                axes[0, 0].set_xticks(range(0, 24, 2))
            else:
                _no_data(axes[0, 0])
        else:
            _no_data(axes[0, 0], 'hour column missing')
        axes[0, 0].set_title('Alerts by Hour of Day')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Alert Count')
        axes[0, 0].grid(True, alpha=0.3)

        day_disp_col = 'day_name' if 'day_name' in self.df.columns else None
        if day_disp_col:
            daily = (self.df[day_disp_col].dropna()
                     .value_counts()
                     .reindex(self.DAY_ORDER, fill_value=0))
            axes[0, 1].bar(range(7), daily.values,
                           color=plt.cm.Set3(np.linspace(0, 1, 7)), edgecolor='black')
            axes[0, 1].set_xticks(range(7))
            axes[0, 1].set_xticklabels(self.SHORT_DAYS)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        else:
            _no_data(axes[0, 1], 'day_name column missing')
        axes[0, 1].set_title('Alerts by Day of Week')
        axes[0, 1].set_ylabel('Alert Count')

        if 'date' in self.df.columns:
            daily_trend = self.df.groupby('date').size()
            if not daily_trend.empty:
                axes[0, 2].plot(daily_trend.index, daily_trend.values,
                                color=self.config.COLORS['accent'], linewidth=2)
                axes[0, 2].fill_between(daily_trend.index, daily_trend.values,
                                        alpha=0.25, color=self.config.COLORS['accent'])
                axes[0, 2].tick_params(axis='x', rotation=40)
                axes[0, 2].grid(True, alpha=0.3)
            else:
                _no_data(axes[0, 2])
        else:
            _no_data(axes[0, 2], 'date column missing')
        axes[0, 2].set_title('Daily Alert Trend')
        axes[0, 2].set_xlabel('Date')
        axes[0, 2].set_ylabel('Alerts')

        day_disp_col2 = 'day_name' if 'day_name' in self.df.columns else None
        if 'hour' in self.df.columns and day_disp_col2:
            tmp   = self.df.dropna(subset=[day_disp_col2, 'hour'])
            pivot = (tmp.groupby([day_disp_col2, 'hour']).size()
                     .unstack(fill_value=0)
                     .reindex(self.DAY_ORDER, fill_value=0))
            if not pivot.empty and pivot.values.sum() > 0:
                sns.heatmap(pivot, cmap='YlOrRd', annot=False,
                            cbar_kws={'label': 'Alert Count'}, ax=axes[1, 0])
            else:
                _no_data(axes[1, 0], 'Not enough data for heatmap')
        else:
            _no_data(axes[1, 0], 'hour or day_name missing')
        axes[1, 0].set_title('Hour × Day Heatmap')
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('Day of Week')

        if 'alert_duration_min' in self.df.columns:
            dur  = _to_float_minutes(self.df['alert_duration_min'].dropna())
            dur  = dur.dropna()
            if not dur.empty:
                q1, q3 = dur.quantile([0.25, 0.75])
                iqr    = q3 - q1
                filt   = dur[(dur >= q1 - 1.5 * iqr) & (dur <= q3 + 1.5 * iqr)]
                if not filt.empty:
                    axes[1, 1].hist(filt, bins=min(50, max(5, len(filt) // 10)),
                                    color=self.config.COLORS['success'],
                                    edgecolor='black', alpha=0.7)
                    axes[1, 1].axvline(dur.mean(), color='red', linestyle='--',
                                       linewidth=2, label=f'Mean: {dur.mean():.1f} min')
                    axes[1, 1].legend()
                else:
                    _no_data(axes[1, 1], 'All durations filtered as outliers')
            else:
                _no_data(axes[1, 1], 'No duration data')
        else:
            _no_data(axes[1, 1], 'alert_duration_min missing')
        axes[1, 1].set_title('Alert Duration Distribution')
        axes[1, 1].set_xlabel('Duration (minutes)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)

        day_disp_col3 = 'day_name' if 'day_name' in self.df.columns else (
                         'is_weekend' if 'is_weekend' in self.df.columns else None)
        if day_disp_col3 == 'day_name':
            dow      = self.df['day_name'].dropna()
            is_we    = dow.isin(['Saturday', 'Sunday'])
            counts   = is_we.value_counts().reindex([False, True], fill_value=0)
            non_zero = counts[counts > 0]
            if not non_zero.empty:
                lbl  = ['Weekday' if not i else 'Weekend' for i in non_zero.index]
                clrs = [self.config.COLORS['primary'] if not i
                        else self.config.COLORS['secondary'] for i in non_zero.index]
                axes[1, 2].pie(non_zero.values, labels=lbl,
                               autopct='%1.1f%%', colors=clrs, startangle=90)
            else:
                _no_data(axes[1, 2])
        elif day_disp_col3 == 'is_weekend':
            counts   = self.df['is_weekend'].value_counts().reindex([0, 1], fill_value=0)
            non_zero = counts[counts > 0]
            if not non_zero.empty:
                lbl  = ['Weekday' if i == 0 else 'Weekend' for i in non_zero.index]
                clrs = [self.config.COLORS['primary'] if i == 0
                        else self.config.COLORS['secondary'] for i in non_zero.index]
                axes[1, 2].pie(non_zero.values, labels=lbl,
                               autopct='%1.1f%%', colors=clrs, startangle=90)
            else:
                _no_data(axes[1, 2])
        else:
            _no_data(axes[1, 2], 'day_name / is_weekend missing')
        axes[1, 2].set_title('Weekday vs Weekend')

        self._show_and_save('temporal_features')

    #  3. Categorical features ─
    def plot_categorical_features(self):
        print("\n Plotting categorical features...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Categorical Feature Analysis', **self.config.FONT_TITLE)

        if 'cause' in self.df.columns:
            cc = self.df['cause'].value_counts().head(10)
            if not cc.empty:
                axes[0, 0].barh(range(len(cc)), cc.values,
                                color=plt.cm.Set3(np.linspace(0, 1, len(cc))))
                axes[0, 0].set_yticks(range(len(cc)))
                axes[0, 0].set_yticklabels(cc.index)
                axes[0, 0].grid(True, alpha=0.3, axis='x')
            else:
                _no_data(axes[0, 0])
        else:
            _no_data(axes[0, 0], 'cause column missing')
        axes[0, 0].set_title('Top 10 Alert Causes')
        axes[0, 0].set_xlabel('Count')

        if 'effect' in self.df.columns:
            ec = self.df['effect'].value_counts().head(10)
            if not ec.empty:
                axes[0, 1].barh(range(len(ec)), ec.values,
                                color=plt.cm.Set2(np.linspace(0, 1, len(ec))))
                axes[0, 1].set_yticks(range(len(ec)))
                axes[0, 1].set_yticklabels(ec.index)
                axes[0, 1].grid(True, alpha=0.3, axis='x')
            else:
                _no_data(axes[0, 1])
        else:
            _no_data(axes[0, 1], 'effect column missing')
        axes[0, 1].set_title('Top 10 Alert Effects')
        axes[0, 1].set_xlabel('Count')

        lang_col = ('language_name' if 'language_name' in self.df.columns else
                    'language'      if 'language'      in self.df.columns else None)
        if lang_col:
            lc = self.df[lang_col].value_counts().head(8)
            if not lc.empty:
                axes[0, 2].pie(lc.values, labels=lc.index, autopct='%1.1f%%',
                               colors=plt.cm.Pastel1(np.linspace(0, 1, len(lc))),
                               startangle=90)
            else:
                _no_data(axes[0, 2])
        else:
            _no_data(axes[0, 2], 'No language column')
        axes[0, 2].set_title('Language Distribution (Top 8)')

        if 'route' in self.df.columns:
            rc = self.df['route'].dropna().value_counts().head(15)
            if not rc.empty:
                axes[1, 0].bar(range(len(rc)), rc.values,
                               color=self.config.COLORS['accent'], edgecolor='black')
                axes[1, 0].set_xticks(range(len(rc)))
                axes[1, 0].set_xticklabels(rc.index, rotation=45, ha='right')
                axes[1, 0].grid(True, alpha=0.3, axis='y')
            else:
                _no_data(axes[1, 0])
        else:
            _no_data(axes[1, 0], 'route column missing')
        axes[1, 0].set_title('Top 15 Routes with Alerts')
        axes[1, 0].set_ylabel('Alert Count')

        if 'cause' in self.df.columns and 'effect' in self.df.columns:
            top_c = self.df['cause'].value_counts().head(5).index
            top_e = self.df['effect'].value_counts().head(5).index
            sub   = self.df[self.df['cause'].isin(top_c) & self.df['effect'].isin(top_e)]
            if not sub.empty:
                pivot = sub.groupby(['cause', 'effect']).size().unstack(fill_value=0)
                if not pivot.empty:
                    sns.heatmap(pivot, annot=True, fmt='d', cmap='YlGnBu',
                                cbar_kws={'label': 'Count'}, ax=axes[1, 1])
                    axes[1, 1].tick_params(axis='x', rotation=30)
                else:
                    _no_data(axes[1, 1], 'Pivot empty')
            else:
                _no_data(axes[1, 1], 'No matching cause/effect rows')
        else:
            _no_data(axes[1, 1], 'cause or effect missing')
        axes[1, 1].set_title('Cause vs Effect (Top 5 each)')

        if 'sentiment' in self.df.columns:
            sc = self.df['sentiment'].loc[self.df['sentiment'] != 'unknown'].value_counts()
            if not sc.empty:
                cm   = {'Positive': '#06A77D', 'Neutral': '#6C757D', 'Negative': '#D62828'}
                clrs = [cm.get(s, self.config.COLORS['neutral']) for s in sc.index]
                axes[1, 2].bar(range(len(sc)), sc.values, color=clrs, edgecolor='black')
                axes[1, 2].set_xticks(range(len(sc)))
                axes[1, 2].set_xticklabels(sc.index)
                axes[1, 2].grid(True, alpha=0.3, axis='y')
            else:
                _no_data(axes[1, 2], 'No sentiment data')
        else:
            _no_data(axes[1, 2], 'sentiment column missing')
        axes[1, 2].set_title('Sentiment Distribution')
        axes[1, 2].set_ylabel('Count')

        self._show_and_save('categorical_features')

    #  4. NLP features ─
    def plot_nlp_features(self):
        print("\n Plotting NLP features...")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('NLP Features Analysis', **self.config.FONT_TITLE)

        if 'sentiment' in self.df.columns and 'language_name' in self.df.columns:
            top_langs = self.df['language_name'].value_counts().head(5).index
            sub = (self.df[self.df['language_name'].isin(top_langs)]
                   .loc[self.df['sentiment'] != 'unknown'])
            if not sub.empty:
                sent_by_lang = (sub.groupby(['language_name', 'sentiment'])
                                .size().unstack(fill_value=0))
                palette     = {'Negative': '#D62828', 'Neutral': '#6C757D', 'Positive': '#06A77D'}
                col_colours = [palette.get(c, '#888888') for c in sent_by_lang.columns]
                sent_by_lang.plot(kind='bar', stacked=False, ax=axes[0, 0], color=col_colours)
                axes[0, 0].legend(title='Sentiment')
                axes[0, 0].tick_params(axis='x', rotation=30)
                axes[0, 0].grid(True, alpha=0.3, axis='y')
            else:
                _no_data(axes[0, 0], 'No language/sentiment data')
        else:
            _no_data(axes[0, 0], 'language_name or sentiment missing')
        axes[0, 0].set_title('Sentiment by Language (Top 5)')
        axes[0, 0].set_xlabel('Language')
        axes[0, 0].set_ylabel('Count')

        if 'topic' in self.df.columns:
            tc = self.df['topic'].dropna().value_counts().head(11)
            tc = tc[tc.index != -1].head(10)
            if not tc.empty:
                axes[0, 1].bar(range(len(tc)), tc.values,
                               color=self.config.COLORS['secondary'], edgecolor='black')
                axes[0, 1].set_xticks(range(len(tc)))
                axes[0, 1].set_xticklabels([f'Topic {int(t)}' for t in tc.index], rotation=30)
                axes[0, 1].grid(True, alpha=0.3, axis='y')
            else:
                _no_data(axes[0, 1], 'No topics (all outliers)')
        else:
            _no_data(axes[0, 1], 'topic column missing')
        axes[0, 1].set_title('Top 10 Topics')
        axes[0, 1].set_ylabel('Alert Count')

        if 'score' in self.df.columns and 'sentiment' in self.df.columns:
            palette = {'Negative': '#D62828', 'Neutral': '#6C757D', 'Positive': '#06A77D'}
            plotted = False
            for sent in self.df['sentiment'].dropna().unique():
                if sent == 'unknown':
                    continue
                sc = self.df[self.df['sentiment'] == sent]['score'].dropna()
                if not sc.empty:
                    axes[1, 0].hist(sc, bins=25, alpha=0.55, label=sent,
                                    edgecolor='white', color=palette.get(sent, '#888888'))
                    plotted = True
            if plotted:
                axes[1, 0].legend()
            else:
                _no_data(axes[1, 0], 'No score data')
        else:
            _no_data(axes[1, 0], 'score or sentiment missing')
        axes[1, 0].set_title('Sentiment Confidence Distribution')
        axes[1, 0].set_xlabel('Confidence Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)

        if 'confidence' in self.df.columns:
            conf = self.df['confidence'].dropna()
            if not conf.empty:
                axes[1, 1].hist(conf, bins=50, color=self.config.COLORS['success'],
                                edgecolor='black', alpha=0.7)
                axes[1, 1].axvline(conf.mean(), color='red', linestyle='--',
                                   linewidth=2, label=f'Mean: {conf.mean():.3f}')
                axes[1, 1].axvline(0.9, color='orange', linestyle=':',
                                   linewidth=2, label='Threshold: 0.9')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                _no_data(axes[1, 1])
        else:
            _no_data(axes[1, 1], 'confidence column missing')
        axes[1, 1].set_title('Language Detection Confidence')
        axes[1, 1].set_xlabel('Confidence')
        axes[1, 1].set_ylabel('Frequency')

        self._show_and_save('nlp_features')

    #  5. Entity distribution
    def plot_entity_distribution(self):
        """Visualize NER entity type counts and top entity texts.

        ADDITION: Corresponds to the MultilingualNER geoparsing output,
        previously produced by analyze_entities() but never visualized.
        """
        print("\n Plotting entity distribution...")
        if 'all_entities' not in self.df.columns:
            print("  No 'all_entities' column found. Run NER first.")
            return

        all_ents  = [e for ents in self.df['all_entities'].dropna() for e in ents]
        if not all_ents:
            print("  No entities to plot.")
            return

        type_counts = Counter(e.get('entity_type', 'UNK') for e in all_ents)
        text_counts = Counter((e.get('entity_text') or '').lower() for e in all_ents)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Named Entity Distribution', **self.config.FONT_TITLE)

        labels = list(type_counts.keys())
        values = list(type_counts.values())
        axes[0].bar(labels, values, color=self.config.COLORS['primary'], edgecolor='black')
        axes[0].set_title('Entity Type Counts')
        axes[0].set_ylabel('Count')
        axes[0].grid(True, alpha=0.3, axis='y')

        top20 = text_counts.most_common(20)
        if top20:
            names, cnts = zip(*top20)
            axes[1].barh(range(len(top20)), cnts, color=self.config.COLORS['secondary'])
            axes[1].set_yticks(range(len(top20)))
            axes[1].set_yticklabels(names)
            axes[1].invert_yaxis()
            axes[1].set_title('Top 20 Entity Texts')
            axes[1].set_xlabel('Count')
            axes[1].grid(True, alpha=0.3, axis='x')

        self._show_and_save('entity_distribution')

    #  6. Geospatial
    def plot_geospatial(self):
        """Plot geocoded alert locations on an interactive map.

        ADDITION: Visualizes the output of MultilingualNER geoparsing —
        the natural geographic complement to the text analytics.
        Requires folium (pip install folium).
        """
        print("\n Plotting geospatial distribution...")
        if 'first_lat' not in self.df.columns or 'first_lon' not in self.df.columns:
            print("  No geocoded location columns found. Run geoparse_to_dataframe() first.")
            return

        geo_df = self.df[['first_lat', 'first_lon', 'first_loc_text',
                           'cause', 'effect']].dropna(subset=['first_lat', 'first_lon'])
        if geo_df.empty:
            print("  No geocoded alerts to plot.")
            return

        try:
            import folium
            center_lat = geo_df['first_lat'].mean()
            center_lon = geo_df['first_lon'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

            for _, row in geo_df.iterrows():
                popup_text = (f"<b>Location:</b> {row.get('first_loc_text', '')}<br>"
                              f"<b>Cause:</b> {row.get('cause', '')}<br>"
                              f"<b>Effect:</b> {row.get('effect', '')}")
                folium.CircleMarker(
                    location=[row['first_lat'], row['first_lon']],
                    radius=5, color=self.config.COLORS['danger'],
                    fill=True, fill_opacity=0.6,
                    popup=folium.Popup(popup_text, max_width=250)
                ).add_to(m)

            path = f'{self.output_dir}/geospatial_alerts.html'
            m.save(path)
            print(f"  ✓ Saved interactive map → {path}")
        except ImportError:
            print("  folium not installed — falling back to static matplotlib scatter plot.")
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.scatter(geo_df['first_lon'], geo_df['first_lat'],
                       alpha=0.4, s=10, color=self.config.COLORS['danger'])
            ax.set_title('Geocoded Alert Locations')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, alpha=0.3)
            self._show_and_save('geospatial_alerts')

    #  7. Correlations ─
    def plot_correlations(self):
        print("\n Plotting correlations...")
        numeric_cols = [
            c for c in self.df.select_dtypes(include=[np.number]).columns
            if not c.endswith('_id')
        ]
        if len(numeric_cols) < 2:
            print("    Not enough numeric columns for correlation")
            return

        corr = self.df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, linewidths=1,
                    cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Feature Correlation Matrix', **self.config.FONT_TITLE)
        plt.tight_layout()
        # was calling self._save() which does not exist; fixed to _show_and_save
        self._show_and_save('correlations')

    #  8. Master dashboard ─
    def create_master_dashboard(self):
        print("\n Creating master dashboard...")
        fig = plt.figure(figsize=(22, 15))
        # replaced non-standard plt.matplotlib.gridspec with imported gridspec
        gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.38, wspace=0.32)
        fig.suptitle('GTFS-RT Alerts — Master Dashboard', **self.config.FONT_TITLE)

        ax1 = fig.add_subplot(gs[0, 0])
        lang_col = ('language_name' if 'language_name' in self.df.columns else
                    'language'      if 'language'      in self.df.columns else None)
        n_langs  = self.df[lang_col].nunique() if lang_col else 0
        if 'date' in self.df.columns:
            try:
                d_max = pd.Timestamp(self.df['date'].max())
                d_min = pd.Timestamp(self.df['date'].min())
                span  = max(0, (d_max - d_min).days)
            except Exception:
                span = 0
        else:
            span = 0
        metrics = {
            'Total Alerts':     len(self.df),
            'Unique Routes':    self.df['route'].nunique() if 'route' in self.df.columns else 0,
            'Languages':        n_langs,
            'Date Span (days)': span,
        }
        y_pos = np.arange(len(metrics))
        ax1.barh(y_pos, list(metrics.values()), color=self.config.COLORS['primary'])
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(list(metrics.keys()))
        ax1.set_title('Key Metrics', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(metrics.values()):
            ax1.text(max(v * 0.02, 0.5), i, f'{v:,}', va='center', fontsize=9)

        ax2 = fig.add_subplot(gs[0, 1:3])
        if 'hour' in self.df.columns:
            hourly = self.df['hour'].dropna().value_counts().sort_index()
            ax2.plot(hourly.index, hourly.values, marker='o', linewidth=2,
                     markersize=6, color=self.config.COLORS['primary'])
            ax2.fill_between(hourly.index, hourly.values, alpha=0.25,
                             color=self.config.COLORS['primary'])
            ax2.set_xticks(range(0, 24, 2))
            ax2.grid(True, alpha=0.3)
        else:
            _no_data(ax2, 'hour column missing')
        ax2.set_title('Alerts by Hour of Day', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Hour')
        ax2.set_ylabel('Count')

        ax3 = fig.add_subplot(gs[0, 3])
        if 'cause' in self.df.columns:
            ct = self.df['cause'].value_counts().head(5)
            if not ct.empty:
                ax3.pie(ct.values, labels=ct.index, autopct='%1.0f%%',
                        startangle=90, textprops={'fontsize': 8})
        else:
            _no_data(ax3, 'cause missing')
        ax3.set_title('Top 5 Causes', fontsize=11, fontweight='bold')

        ax4 = fig.add_subplot(gs[1, :2])
        if 'date' in self.df.columns:
            daily = self.df.groupby('date').size()
            if not daily.empty:
                ax4.plot(daily.index, daily.values, linewidth=2,
                         color=self.config.COLORS['accent'])
                ax4.fill_between(daily.index, daily.values, alpha=0.25,
                                 color=self.config.COLORS['accent'])
                ax4.tick_params(axis='x', rotation=30)
                ax4.grid(True, alpha=0.3)
            else:
                _no_data(ax4)
        else:
            _no_data(ax4, 'date column missing')
        ax4.set_title('Daily Alert Trend', fontsize=11, fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Alerts')

        ax5 = fig.add_subplot(gs[1, 2:])
        if lang_col:
            lt = self.df[lang_col].value_counts().head(8)
            if not lt.empty:
                ax5.barh(range(len(lt)), lt.values, color=self.config.COLORS['primary'])
                ax5.set_yticks(range(len(lt)))
                ax5.set_yticklabels(lt.index)
                ax5.grid(True, alpha=0.3, axis='x')
            else:
                _no_data(ax5)
        else:
            _no_data(ax5, 'No language column')
        ax5.set_title('Top Languages', fontsize=11, fontweight='bold')

        ax6 = fig.add_subplot(gs[2, :2])
        if 'effect' in self.df.columns:
            et = self.df['effect'].value_counts().head(8)
            if not et.empty:
                ax6.barh(range(len(et)), et.values, color=self.config.COLORS['secondary'])
                ax6.set_yticks(range(len(et)))
                ax6.set_yticklabels(et.index)
                ax6.grid(True, alpha=0.3, axis='x')
            else:
                _no_data(ax6)
        else:
            _no_data(ax6, 'effect missing')
        ax6.set_title('Top Alert Effects', fontsize=11, fontweight='bold')

        ax7 = fig.add_subplot(gs[2, 2])
        if 'sentiment' in self.df.columns:
            sc = (self.df['sentiment'].loc[self.df['sentiment'] != 'unknown'].value_counts())
            if not sc.empty:
                cm = {'Positive': '#06A77D', 'Neutral': '#6C757D', 'Negative': '#D62828'}
                ax7.pie(sc.values, labels=sc.index, autopct='%1.0f%%',
                        colors=[cm.get(s, '#6C757D') for s in sc.index],
                        startangle=90, textprops={'fontsize': 9})
            else:
                _no_data(ax7, 'No sentiment data')
        else:
            _no_data(ax7, 'sentiment missing')
        ax7.set_title('Sentiment', fontsize=11, fontweight='bold')

        ax8 = fig.add_subplot(gs[2, 3])
        if 'text_length' in self.df.columns:
            tl = self.df['text_length'].dropna()
            if not tl.empty:
                ax8.hist(tl, bins=30, edgecolor='black', alpha=0.7,
                         color=self.config.COLORS['accent'])
                ax8.axvline(tl.mean(), color='red', linestyle='--',
                            linewidth=2, label=f'Mean: {tl.mean():.0f}')
                ax8.legend(fontsize=8)
                ax8.grid(True, alpha=0.3)
            else:
                _no_data(ax8)
        else:
            _no_data(ax8, 'text_length missing')
        ax8.set_title('Text Length', fontsize=11, fontweight='bold')
        ax8.set_xlabel('Characters')

        self._show_and_save('master_dashboard')

    def generate_all_visualizations(self):
        print("\n" + "=" * 80)
        print(" GENERATING COMPLETE VISUALIZATION SUITE")
        print("=" * 80)
        errors = []
        for name, fn in [
            ('Text Feature Analysis',         self.plot_text_features),
            ('Temporal Feature Analysis',      self.plot_temporal_features),
            ('NLP Feature Analysis',           self.plot_nlp_features),
            ('Entity Distribution',            self.plot_entity_distribution),
            ('Geospatial',                     self.plot_geospatial),
            ('GTFS Alerts — Master Dashboard', self.create_master_dashboard),
            ('Categorical Feature Analysis',   self.plot_categorical_features),
        ]:
            try:
                fn()
            except Exception as exc:
                import traceback as tb
                print(f"\n {name}: {exc}")
                tb.print_exc()
                errors.append(name)

        print("\n" + "=" * 80)
        print("  Errors in: " + ", ".join(errors) if errors else "  ALL COMPLETE")
        files = sorted(f for f in os.listdir(self.output_dir) if f.endswith('.png'))
        print(f"\n {self.output_dir}/ — {len(files)} file(s)")
        for f in files:
            print(f"   • {f}")


# SECTION 8: EXPORT

class ResultsExporter:
    """Export analysis results in multiple formats."""

    # Columns that contain Python lists/dicts and cannot be round-tripped via CSV or Parquet
    _LIST_COLS = ['active_periods', 'informed_entities', 'all_entities', 'loc_entities']

    def _flat_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with known list/dict columns dropped for tabular export."""
        drop = [c for c in self._LIST_COLS if c in df.columns]
        return df.drop(columns=drop) if drop else df.copy()

    def export_analysis_results(self, df, output_dir: str = 'output'):
        if df is None or df.empty:
            print("No data to export")
            return
        os.makedirs(output_dir, exist_ok=True)

        flat = self._flat_df(df)
        flat.to_csv(f'{output_dir}/processed_alerts.csv', index=False)
        print(f"✓ Exported processed data to {output_dir}/processed_alerts.csv")

        self._export_summary(flat, output_dir)
        self._export_cause_distribution(flat, output_dir)
        self._export_route_statistics(flat, output_dir)
        self.export_to_parquet(flat, output_dir)   # export_to_parquet still self-filters via PyArrow probe

        print(f"\n✓ All results exported to '{output_dir}/'")

    def _export_summary(self, df, output_dir):
        summary = {
            'total_alerts':     len(df),
            'unique_routes':    df['route'].nunique() if 'route' in df.columns else None,
            'avg_duration_min': df['alert_duration_min'].mean()
                                if 'alert_duration_min' in df.columns else None,
            'avg_text_length':  df['text_length'].mean()
                                if 'text_length' in df.columns else None,
            'date_range':       (f"{df['id_time'].min()} to {df['id_time'].max()}"
                                if 'id_time' in df.columns else None),
        }
        pd.DataFrame([summary]).to_csv(f'{output_dir}/summary_statistics.csv', index=False)
        print(f"✓ Exported summary to {output_dir}/summary_statistics.csv")

    def _export_cause_distribution(self, df, output_dir):
        if 'cause' not in df.columns:
            return
        cause_dist = df['cause'].value_counts().reset_index()
        cause_dist.columns = ['cause', 'count']
        cause_dist.to_csv(f'{output_dir}/cause_distribution.csv', index=False)
        print(f"✓ Exported cause distribution to {output_dir}/cause_distribution.csv")

    def _export_route_statistics(self, df, output_dir):
        """Export per-route statistics.

        REMOVAL: Replaced dynamic column-naming pattern with explicit rename()
        to be robust against column ordering changes in future pandas versions.
        """
        if 'route' not in df.columns:
            return

        agg_dict = {'alert_id': 'count'}
        if 'alert_duration_min' in df.columns:
            agg_dict['alert_duration_min'] = 'mean'
        if 'text_length' in df.columns:
            agg_dict['text_length'] = 'mean'

        route_stats = df.groupby('route').agg(agg_dict).reset_index()

        rename_map = {'alert_id': 'alert_count'}
        if 'alert_duration_min' in route_stats.columns:
            rename_map['alert_duration_min'] = 'avg_duration_min'
        if 'text_length' in route_stats.columns:
            rename_map['text_length'] = 'avg_text_length'
        route_stats = route_stats.rename(columns=rename_map)

        route_stats.to_csv(f'{output_dir}/route_statistics.csv', index=False)
        print(f"✓ Exported route statistics to {output_dir}/route_statistics.csv")

    def export_to_parquet(self, df: pd.DataFrame, output_dir: str = 'output',
                           filename: str = 'processed_alerts.parquet') -> None:
        """Export processed alerts to Parquet format.

        Robustly filters out columns that PyArrow cannot serialize:
          - list/dict-valued object columns
          - mixed-type object columns (e.g. day_of_week written as both int and str
            by different pipeline stages before the column contract was enforced)
          - timedelta columns (converted to float seconds instead)
        All other dtypes — datetime64, numeric, bool, pure-string object — are kept.
        """
        import pyarrow as pa

        os.makedirs(output_dir, exist_ok=True)
        path = f'{output_dir}/{filename}'

        exportable = df.copy()

        # Convert timedelta to float seconds (Parquet has no timedelta type)
        for col in exportable.select_dtypes(include='timedelta64[ns]').columns:
            exportable[col] = exportable[col].dt.total_seconds()

        # For object columns, probe each one with PyArrow and drop failures
        bad_cols = []
        for col in exportable.select_dtypes(include='object').columns:
            series = exportable[col]
            # Fast check: any list or dict values -> definitely not serializable
            if series.dropna().apply(lambda x: isinstance(x, (list, dict))).any():
                bad_cols.append(col)
                continue
            # Try arrow inference; catches mixed int/str and other surprises
            try:
                pa.array(series, from_pandas=True)
            except (pa.ArrowTypeError, pa.ArrowInvalid):
                bad_cols.append(col)

        if bad_cols:
            print(f"  Parquet: dropping {len(bad_cols)} non-serializable column(s): {bad_cols}")
            exportable = exportable.drop(columns=bad_cols)

        exportable.to_parquet(path, index=False)
        print(f"Exported Parquet to {path} "
              f"({len(exportable.columns)} columns, {len(exportable):,} rows)")

    def export_nlp_results(self, df: pd.DataFrame, output_dir: str = 'output') -> None:
        """Export NLP-specific columns (language, sentiment, entities) separately.

        ADDITION: NLP columns contain nested structures (lists of dicts)
        that CSV handles poorly. JSON is the correct format for these fields.
        """
        import json
        os.makedirs(output_dir, exist_ok=True)
        nlp_cols = [c for c in ['alert_id', 'language_code', 'language_name', 'confidence',
                                  'sentiment', 'score', 'all_entities', 'loc_entities',
                                  'first_loc_text', 'first_lat', 'first_lon']
                    if c in df.columns]
        if not nlp_cols:
            print("  No NLP result columns found to export.")
            return
        nlp_df = df[nlp_cols].copy()
        path   = f'{output_dir}/nlp_results.json'
        nlp_df.to_json(path, orient='records', indent=2)
        print(f"✓ Exported NLP results to {path}")

    def export_geojson(self, df: pd.DataFrame, output_dir: str = 'output') -> None:
        """Export geocoded alerts as GeoJSON for direct GIS tool import.

        ADDITION: Standard format for geographic alert data; enables loading
        into QGIS, Mapbox, Leaflet, ArcGIS, etc.
        """
        import json
        if 'first_lat' not in df.columns or 'first_lon' not in df.columns:
            print("  No geocoded columns found. Run geoparse_to_dataframe() first.")
            return

        geo_df = df.dropna(subset=['first_lat', 'first_lon'])
        if geo_df.empty:
            print("  No geocoded rows to export.")
            return

        features = []
        for _, row in geo_df.iterrows():
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row['first_lon'], row['first_lat']],
                },
                "properties": {
                    k: v for k, v in row.items()
                    if k not in ('first_lat', 'first_lon', 'all_entities', 'loc_entities')
                    and not isinstance(v, list)
                    and (pd.notna(v) if not isinstance(v, list) else True)
                }
            }
            features.append(feature)

        geojson = {"type": "FeatureCollection", "features": features}
        os.makedirs(output_dir, exist_ok=True)
        path = f'{output_dir}/alerts_geocoded.geojson'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2, default=str)
        print(f"✓ Exported GeoJSON to {path}")

# geoparse_to_dataframe
# SECTION 9: LANGUAGE ANALYZER

class LanguageAnalyzer:
    """Dedicated language analysis."""

    def __init__(self, output_dir: str = 'visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def analyze_distribution(self, df: pd.DataFrame,
                              lang_col: str = 'language_name',
                              save_plots: bool = True) -> pd.DataFrame:
        if lang_col not in df.columns:
            print(f"Column '{lang_col}' not found")
            return pd.DataFrame()

        print("\n" + "=" * 60)
        print("LANGUAGE DISTRIBUTION ANALYSIS")
        print("=" * 60)

        lang_counts = df[lang_col].value_counts()
        total       = len(df)
        print(f"\nTotal records: {total:,}")
        print(f"\nLanguage Distribution:")
        for lang, count in lang_counts.items():
            pct = (count / total) * 100
            print(f"  {lang}: {count:,} ({pct:.2f}%)")

        summary_df = lang_counts.reset_index()
        summary_df.columns = ['language', 'count']
        summary_df['percentage'] = (summary_df['count'] / total * 100).round(2)

        if save_plots:
            self.plot_language_distribution(df, lang_col)

        return summary_df
#geoparse_to_dataframe
    def plot_language_distribution(self, df: pd.DataFrame,
                                    lang_col: str = 'language_name'):
        if lang_col not in df.columns:
            print(f"Column '{lang_col}' not found")
            return

        lang_counts = df[lang_col].value_counts().head(10)
        if lang_counts.empty:
            print("  No language data to plot")
            return

        n   = len(lang_counts)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Language Distribution', fontsize=14, fontweight='bold')

        axes[0].barh(range(n), lang_counts.values,
                     color=plt.cm.viridis(np.linspace(0, 1, n)))
        axes[0].set_yticks(range(n))
        axes[0].set_yticklabels(lang_counts.index)
        axes[0].set_title('Top 10 Languages (count)')
        axes[0].set_xlabel('Number of Alerts')
        axes[0].grid(True, alpha=0.3, axis='x')

        axes[1].pie(lang_counts.values, labels=lang_counts.index, autopct='%1.1f%%',
                    colors=plt.cm.Pastel1(np.linspace(0, 1, n)), startangle=90)
        axes[1].set_title('Language Share (Top 10)')

        plt.tight_layout()
        plt.show()

        path = f'{self.output_dir}/language_distribution.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f"✓ Saved → {path}")

    def calculate_confidence_stats(self, df: pd.DataFrame,
                                    confidence_col: str = 'confidence') -> Dict:
        if confidence_col not in df.columns:
            print(f"Column '{confidence_col}' not found")
            return {}

        conf  = df[confidence_col].dropna()
        total = len(conf)
        stats = {
            'mean':                float(conf.mean()),
            'median':              float(conf.median()),
            'std':                 float(conf.std()),
            'low_confidence':      int((conf < 0.9).sum()),
            'high_confidence':     int((conf > 0.95).sum()),
            'low_confidence_pct':  float((conf < 0.9).sum() / total * 100),
            'high_confidence_pct': float((conf > 0.95).sum() / total * 100),
        }

        print(f"\nConfidence Statistics:")
        for k, v in stats.items():
            fmt = (f'{v:.2f}%' if 'pct' in k else
                   f'{v:.4f}' if isinstance(v, float) else f'{v:,}')
            print(f"  {k}: {fmt}")
        return stats

    def identify_unknown_languages(self, df: pd.DataFrame,
                                    lang_col: str = 'language_name') -> pd.DataFrame:
        if lang_col not in df.columns:
            return pd.DataFrame()
        mask   = df[lang_col].isin(['unknown', 'UNKNOWN']) | df[lang_col].isna()
        unk_df = df[mask].copy()
        print(f"\nUnknown-language alerts: {len(unk_df):,} "
              f"({len(unk_df)/len(df)*100:.2f}%)")
        if 'text_length' in unk_df.columns and not unk_df.empty:
            print(f"  Avg text length: {unk_df['text_length'].mean():.1f} chars")
        return unk_df

    def export_summary(self, df: pd.DataFrame,
                        filename: str = 'language_summary.csv',
                        lang_col: str = 'language_name') -> str:
        summary = self.analyze_distribution(df, lang_col, save_plots=False)
        if summary.empty:
            return ''
        path = os.path.join(self.output_dir, filename)
        summary.to_csv(path, index=False)
        print(f"✓ Language summary saved → {path}")
        return path

# geoparse_to_dataframe
# SECTION 10: TEXT EDA FUNCTIONS

def perform_text_eda(alerts_df: pd.DataFrame,
                     text_column: str = 'description_text',
                     max_display: int = 50,
                     save_visualizations: bool = True) -> Dict:
    """Standalone comprehensive text EDA."""
    print("\n" + "=" * 80)
    print(f"TEXT COLUMN EDA: {text_column.upper()}")
    print("=" * 80)

    viz_dir = 'visualizations'
    os.makedirs(viz_dir, exist_ok=True)
    df = alerts_df.copy()

    if text_column not in df.columns:
        print(f"Column '{text_column}' not found")
        return {}

    missing_count = df[text_column].isna().sum()
    total_count   = len(df)
    print(f"\n Total records:    {total_count:,}")
    print(f" Missing values:   {missing_count:,} ({missing_count/total_count*100:.1f}%)")

    text_data   = df[text_column].dropna()
    valid_count = len(text_data)
    if valid_count == 0:
        print("No valid text data")
        return {}

    print(f"✓ Valid records:   {valid_count:,}")

    text_lengths = text_data.str.len()
    word_counts  = text_data.str.split().str.len()

    print(f"\n Character Length — min:{text_lengths.min():.0f}  "
          f"max:{text_lengths.max():.0f}  "
          f"mean:{text_lengths.mean():.1f}  "
          f"median:{text_lengths.median():.1f}")
    print(f" Word Count       — min:{word_counts.min():.0f}  "
          f"max:{word_counts.max():.0f}  "
          f"mean:{word_counts.mean():.1f}  "
          f"median:{word_counts.median():.1f}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Text Length Analysis: {text_column}', fontsize=16, fontweight='bold')

    axes[0, 0].hist(text_lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(text_lengths.mean(), color='red', linestyle='--',
                       label=f'Mean: {text_lengths.mean():.1f}')
    axes[0, 0].axvline(text_lengths.median(), color='green', linestyle='--',
                       label=f'Median: {text_lengths.median():.1f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(word_counts, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Word Count Distribution')
    axes[0, 1].set_xlabel('Words')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].boxplot(text_lengths, vert=False, patch_artist=True,
                       boxprops=dict(facecolor='lightcoral'))
    axes[0, 2].set_title('Character Length Box Plot')
    axes[0, 2].set_xlabel('Characters')
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].hist(text_lengths, bins=50, color='gold', edgecolor='black', alpha=0.7, log=True)
    axes[1, 0].set_title('Character Length (Log Scale)')
    axes[1, 0].set_xlabel('Characters')
    axes[1, 0].set_ylabel('Frequency (Log)')
    axes[1, 0].grid(True, alpha=0.3)

    sorted_len = np.sort(text_lengths)
    y_vals     = np.arange(1, len(sorted_len) + 1) / len(sorted_len)
    axes[1, 1].plot(sorted_len, y_vals * 100, color='purple', linewidth=2)
    axes[1, 1].axhline(50, color='red',   linestyle='--', alpha=0.5, label='50%')
    axes[1, 1].axhline(90, color='green', linestyle='--', alpha=0.5, label='90%')
    axes[1, 1].set_title('Cumulative Distribution')
    axes[1, 1].set_xlabel('Characters')
    axes[1, 1].set_ylabel('Percentage (%)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].scatter(word_counts, text_lengths, alpha=0.5, color='teal', s=10)
    if len(word_counts) > 1:
        z = np.polyfit(word_counts, text_lengths, 1)
        p = np.poly1d(z)
        axes[1, 2].plot(word_counts, p(word_counts), "r--", alpha=0.8,
                        label=f'y = {z[0]:.1f}x + {z[1]:.1f}')
        axes[1, 2].legend()
    axes[1, 2].set_title('Word Count vs Character Length')
    axes[1, 2].set_xlabel('Word Count')
    axes[1, 2].set_ylabel('Characters')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_visualizations:
        save_path = f'{viz_dir}/text_length_analysis_{text_column}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    plt.show()

    dup_counts = text_data.value_counts()
    exact_dups = dup_counts[dup_counts > 1]
    print(f"\n Exact duplicates: {len(exact_dups):,}")
    if len(exact_dups) > 0:
        for text, count in exact_dups.head(5).items():
            snippet = text[:80] + ('...' if len(text) > 80 else '')
            print(f"  '{snippet}' → {count:,}×")

    summary_stats = {
        'total_records':        total_count,
        'missing_text':         missing_count,
        'missing_percentage':   missing_count / total_count * 100,
        'avg_text_length':      text_lengths.mean(),
        'median_text_length':   text_lengths.median(),
        'avg_word_count':       word_counts.mean(),
        'exact_duplicates':     len(exact_dups),
        'duplicate_percentage': (exact_dups.sum() / len(text_data) * 100
                                 if len(exact_dups) > 0 else 0),
    }
    return summary_stats


def generate_additional_text_visualizations(alerts_df: pd.DataFrame,
                                             text_column: str = 'description_text',
                                             save_dir: str = 'visualizations'):
    """Word cloud and additional text visuals."""
    if text_column not in alerts_df.columns:
        return
    text_data = alerts_df[text_column].dropna()
    if len(text_data) == 0:
        return
    os.makedirs(save_dir, exist_ok=True)
    try:
        from wordcloud import WordCloud
        all_text  = ' '.join(text_data.astype(str).tolist())
        wordcloud = WordCloud(
            width=800, height=400, background_color='white',
            max_words=100, colormap='viridis'
        ).generate(all_text)
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Alert Descriptions', fontsize=16, fontweight='bold')
        path = f'{save_dir}/wordcloud_{text_column}.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Word cloud saved: {path}")
    except ImportError:
        print("  WordCloud not installed: pip install wordcloud")
    except Exception as e:
        print(f"  Error generating word cloud: {e}")


# SECTION 11: FEATURE SELECTOR

def check_dataframe(df: pd.DataFrame, label: str = "DataFrame") -> None:
    """Run pre-fit diagnostics to catch dtype/overflow issues early.

    REMOVAL: Was incorrectly nested as a free function inside the class body.
    Moved to module level as a standalone utility function.
    """
    FLOAT32_MAX = np.finfo(np.float32).max
    print(f"\n {label} checks ")

    inf_cols = [c for c in df.select_dtypes(include='number').columns
                if np.isinf(df[c].replace(pd.NaT, np.nan)).any()]
    print(f"  Inf columns  ({len(inf_cols)}): {inf_cols}")

    td_cols  = df.select_dtypes(include='timedelta64[ns]').columns
    nat_cols = [c for c in td_cols if df[c].isna().any()]
    print(f"  NaT timedelta columns ({len(nat_cols)}): {nat_cols}")

    large_cols = [c for c in df.select_dtypes(include='number').columns
                  if (df[c].abs() > FLOAT32_MAX).any()]
    print(f"  Overflow columns ({len(large_cols)}): {large_cols}")

    nan_counts = df.isnull().sum()
    nan_cols   = nan_counts[nan_counts > 0]
    print(f"  NaN columns  ({len(nan_cols)}):\n{nan_cols.to_string()}\n")


class GTFSStreamingFeatureSelector:
    """Streaming GTFS feature selection and disruption prediction."""

    _DEFAULT_DROP_COLS = [
        'trip_id', 'vehicle_id', 'timestamp', 'route_id', 'stop_id',
        'alert_id', 'description_text', 'clean_text', 'combined_text',
        'rt_id', 'date', 'id_time',
        'direction_id_x', 'id_date_x', 'RT_id_x',
        'direction_id_y', 'id_date_y', 'RT_id_y',
    ]

    def __init__(self,
                 target:             str                  = 'early_warning_flag',
                 problem_type:       str                  = 'classification',
                 top_mi:             int                  = 30,
                 corr_threshold:     float                = 0.90,
                 variance_threshold: float                = 0.01,
                 n_estimators:       int                  = 80,
                 max_depth:          int                  = 10,
                 min_samples_leaf:   int                  = 10,
                 max_features:       float                = 0.5,
                 rfecv_step:         float                = 0.1,
                 rfecv_cv:           int                  = 3,
                 rfecv_min_features: int                  = 5,
                 rfecv_sample_size:  int                  = 50_000,
                 test_size:          float                = 0.2,
                 random_state:       int                  = 42,
                 drop_cols:          Optional[List[str]]  = None):

        if problem_type not in ('classification', 'regression'):
            raise ValueError("problem_type must be 'classification' or 'regression'")
        if not (0 < corr_threshold <= 1):
            raise ValueError("corr_threshold must be in (0, 1]")

        self.target             = target
        self.problem_type       = problem_type
        self.top_mi             = top_mi
        self.corr_threshold     = corr_threshold
        self.variance_threshold = variance_threshold
        self.n_estimators       = n_estimators
        self.max_depth          = max_depth
        self.min_samples_leaf   = min_samples_leaf
        self.max_features       = max_features
        self.rfecv_step         = rfecv_step
        self.rfecv_cv           = rfecv_cv
        self.rfecv_min_features = rfecv_min_features
        self.rfecv_sample_size  = rfecv_sample_size
        self.test_size          = test_size
        self.random_state       = random_state
        self.drop_cols          = drop_cols or self._DEFAULT_DROP_COLS

        self.model_             = None
        self.rfecv_             = None
        self.selected_features_ = None
        self.X_final_           = None
        self.X_test_            = None
        self.y_test_            = None
        self.mi_scores_         = None
        self.dropped_corr_      = None
        self.dropped_var_       = None
        self._is_fitted         = False
        self._cat_encodings_    = {}
        self._dropped_cat_cols_ = []

    #  Private helpers ─

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call fit() before using this method.")

    def _build_model(self):
        kwargs = dict(
            n_estimators     = self.n_estimators,
            max_depth        = self.max_depth,
            min_samples_leaf = self.min_samples_leaf,
            max_features     = self.max_features,
            n_jobs           = -1,
            random_state     = self.random_state,
        )
        return (RandomForestClassifier(**kwargs) if self.problem_type == 'classification'
                else RandomForestRegressor(**kwargs))

    def _get_numeric_X(self, df: pd.DataFrame, store_encoding: bool = False) -> pd.DataFrame:
        """Convert DataFrame to a numeric-only feature matrix.

        REMOVAL: Removed duplicate timedelta/boolean conversion block that
        followed an early return statement and was therefore unreachable code.
        """
        FLOAT32_MAX = np.finfo(np.float32).max
        df = df.copy()

        for col in df.select_dtypes(include=['object', 'category']).columns:
            if df[col].apply(lambda x: isinstance(x, list)).any():
                df[col] = df[col].apply(
                    lambda x: '|'.join(map(str, x)) if isinstance(x, list) else x
                )
                print(f"  ℹ Flattened list values in '{col}'")

            if store_encoding:
                cats = pd.Categorical(df[col]).categories
                if len(cats) <= 50:
                    self._cat_encodings_[col] = cats
                    df[col] = pd.Categorical(df[col], categories=cats).codes
                    print(f"  ℹ Encoded: '{col}' ({len(cats)} categories)")
                else:
                    self._dropped_cat_cols_.append(col)
                    print(f"  ✂ Dropped high-cardinality: '{col}'")
            else:
                if col in self._cat_encodings_:
                    df[col] = pd.Categorical(
                        df[col], categories=self._cat_encodings_[col]
                    ).codes

        df = df.drop(columns=self._dropped_cat_cols_, errors='ignore')

        # Timedelta → seconds
        for col in df.select_dtypes(include='timedelta64[ns]').columns:
            df[col] = df[col].dt.total_seconds()

        # Boolean → int
        for col in df.select_dtypes(include='bool').columns:
            df[col] = df[col].astype(int)

        return (df
                .select_dtypes(include='number')
                .replace([np.inf, -np.inf], np.nan)
                .astype(float)
                .pipe(lambda d: d.fillna(d.median()))
                .pipe(lambda d: d.dropna(axis=1))
                .clip(-FLOAT32_MAX, FLOAT32_MAX))

    def _remove_correlated(self, X: pd.DataFrame) -> pd.DataFrame:
        upper   = X.corr().abs().where(
            np.triu(np.ones((len(X.columns),) * 2), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > self.corr_threshold)]
        self.dropped_corr_ = to_drop
        print(f"  Removed {len(to_drop)} correlated features (threshold={self.corr_threshold})")
        return X.drop(columns=to_drop)

    def _mutual_information(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        fn = (mutual_info_classif if self.problem_type == 'classification'
              else mutual_info_regression)
        return pd.Series(
            fn(X, y, random_state=self.random_state), index=X.columns
        ).sort_values(ascending=False)

    #  Public: training pipeline ─

    def fit(self, df: pd.DataFrame) -> 'GTFSStreamingFeatureSelector':
        from sklearn.feature_selection import VarianceThreshold, RFECV
        from sklearn.model_selection import train_test_split, KFold

        print("=" * 60)
        print("GTFS STREAMING FEATURE SELECTION PIPELINE")
        print("=" * 60)

        if self.target not in df.columns:
            raise ValueError(f"Target '{self.target}' not found. "
                             f"Available: {list(df.columns)}")

        df_model = (df
                    .drop(columns=self.drop_cols, errors='ignore')
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna(thresh=len(df) * 0.6, axis=1)
                    .pipe(lambda d: d.fillna(d.median(numeric_only=True))))

        y = df_model[self.target]
        X = self._get_numeric_X(df_model.drop(columns=[self.target]), store_encoding=True)

        stratify = y if self.problem_type == 'classification' else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=stratify,
            random_state=self.random_state
        )
        self.X_test_ = X_test
        self.y_test_ = y_test

        print("\n[1/5] Correlation filtering...")
        X_train = self._remove_correlated(X_train)
        X_test  = X_test[X_train.columns]

        print(f"\n[2/5] Variance threshold (threshold={self.variance_threshold})...")
        var_sel  = VarianceThreshold(self.variance_threshold)
        X_var    = var_sel.fit_transform(X_train)
        retained = X_train.columns[var_sel.get_support()]
        self.dropped_var_ = list(X_train.columns[~var_sel.get_support()])
        X_train  = pd.DataFrame(X_var, columns=retained, index=X_train.index)
        X_test   = X_test[retained]
        print(f"  Removed {len(self.dropped_var_)} low-variance features")

        print(f"\n[3/5] Mutual information filtering (top {self.top_mi})...")
        self.mi_scores_ = self._mutual_information(X_train, y_train)
        top_features    = self.mi_scores_.head(self.top_mi).index
        X_train = X_train[top_features]
        X_test  = X_test[top_features]
        print(f"  Kept top {self.top_mi} features by MI score")

        print(f"\n[4/5] RFECV feature selection (cv={self.rfecv_cv}, step={self.rfecv_step})...")
        n_sample  = min(self.rfecv_sample_size, len(X_train))
        X_sample  = X_train.reset_index(drop=True).sample(n=n_sample,
                                                           random_state=self.random_state)
        y_sample  = y_train.reset_index(drop=True).loc[X_sample.index]

        scoring   = ("neg_root_mean_squared_error" if self.problem_type == 'regression'
                     else "f1_weighted")
        self.rfecv_ = RFECV(
            estimator              = self._build_model(),
            step                   = self.rfecv_step,
            cv                     = KFold(self.rfecv_cv, shuffle=True,
                                           random_state=self.random_state),
            scoring                = scoring,
            min_features_to_select = self.rfecv_min_features,
            n_jobs                 = -1,
        )
        self.rfecv_.fit(X_sample, y_sample)

        self.selected_features_ = list(X_sample.columns[self.rfecv_.support_])
        print(f"  Optimal features: {len(self.selected_features_)}")
        print(f"  Selected: {self.selected_features_}")

        self.X_final_ = X_train[self.selected_features_]

        print(f"\n[5/5] Final model fit on {len(self.selected_features_)} features...")
        self.model_ = self._build_model()
        self.model_.fit(self.X_final_, y_train)

        self._is_fitted = True
        print("\n✓ Pipeline complete!")
        return self

    #  Public: inference ─

    def predict_batch(self, df_batch: pd.DataFrame) -> np.ndarray:
        """Predict on a new batch using the selected features."""
        self._check_fitted()
        X = (df_batch
             .drop(columns=self.drop_cols + [self.target], errors='ignore')
             .replace([np.inf, -np.inf], np.nan)
             .pipe(lambda d: d.fillna(d.median(numeric_only=True)))
             .pipe(self._get_numeric_X)
             .reindex(columns=self.selected_features_, fill_value=0))
        return self.model_.predict(X)

    #  Public: explainability

    def explain(self, max_display: int = 20) -> None:
        """Generate SHAP summary plot for the selected features."""
        self._check_fitted()
        print("\nGenerating SHAP summary plot...")
        explainer   = shap.TreeExplainer(self.model_)
        shap_values = explainer(self.X_final_)
        shap.summary_plot(shap_values.values, self.X_final_, max_display=max_display)

    #  Public: reporting ─

    def summary(self) -> Dict:
        """Return dict summarising all pipeline stages."""
        self._check_fitted()
        return {
            'selected_features':    self.selected_features_,
            'n_selected':           len(self.selected_features_),
            'dropped_correlated':   self.dropped_corr_,
            'dropped_low_variance': self.dropped_var_,
            'top_mi_scores':        self.mi_scores_.head(self.top_mi).to_dict(),
            'optimal_n_features':   self.rfecv_.n_features_,
        }

    #  Public: diagnostic

    def diagnose(self, df: pd.DataFrame) -> None:
        """Run pre-fit diagnostics to catch dtype/overflow issues early."""
        FLOAT32_MAX = np.finfo(np.float32).max
        print("=" * 60)
        print("DATAFRAME DIAGNOSTIC")
        print("=" * 60)

        td_cols = df.select_dtypes(include='timedelta64[ns]').columns
        print(f"\n[1] Timedelta columns ({len(td_cols)}):")
        for col in td_cols:
            print(f"    {col}: {df[col].isna().sum()} NaT | max={df[col].dropna().max()}")

        df2 = df.copy()
        for col in td_cols:
            df2[col] = df2[col].dt.total_seconds()

        num = df2.select_dtypes(include='number').replace([np.inf, -np.inf], np.nan)

        inf_cols = {c: np.isinf(df2[c]).sum() for c in num.columns
                    if np.isinf(df2.get(c, pd.Series(dtype=float))).sum() > 0}
        print(f"\n[2] Inf columns ({len(inf_cols)}): {list(inf_cols.keys())}")

        large = {c: num[c].abs().max() for c in num.columns
                 if num[c].abs().max() > FLOAT32_MAX}
        print(f"\n[3] Overflow columns ({len(large)}):")
        for c, v in large.items():
            print(f"    {c}: {v:.4e}")

        nan_counts = num.isnull().sum()
        nan_cols   = nan_counts[nan_counts > 0]
        print(f"\n[4] NaN columns ({len(nan_cols)}):")
        print(nan_cols.to_string() if not nan_cols.empty else "    None")

        list_cols = [c for c in df.select_dtypes(include='object').columns
                     if df[c].apply(lambda x: isinstance(x, list)).any()]
        print(f"\n[5] List-valued columns — will be flattened ({len(list_cols)}): {list_cols}")
        for col in list_cols:
            sample = df[col].dropna().iloc[0]
            print(f"    {col}: e.g. {sample}")
        print("=" * 60)


# SECTION 12: PIPELINE ORCHESTRATORS

class GTFSPipeline:
    """Orchestrates the full GTFS disruption analysis pipeline."""

    def __init__(self,
                 ner_model:        str  = "Davlan/xlm-roberta-base-ner-hrl",
                 enable_geocoding: bool = True,
                 target:           str  = 'Is_Disruption',
                 problem_type:     str  = 'classification',
                 top_n:            int  = 15):

        self.ner      = MultilingualNER(model_name=ner_model, enable_geocoding=enable_geocoding)
        self.selector = GTFSStreamingFeatureSelector(
            target=target, problem_type=problem_type, top_mi=top_n)

    def run(self, df: pd.DataFrame, text_column: str = 'combined_text') -> pd.DataFrame:
        """Full pipeline: geoparse → feature selection → predict.

        ADDITION: Checks for text_column existence and falls back to
        description_text to avoid ValueError from geoparse_to_dataframe.
        """
        # existence check with fallback
        if text_column not in df.columns:
            fallback = 'description_text'
            if fallback in df.columns:
                print(f"  Warning: '{text_column}' not found. "
                      f"Falling back to '{fallback}'.")
                text_column = fallback
            else:
                raise ValueError(
                    f"Neither '{text_column}' nor 'description_text' found in DataFrame.")

        print("\n Step 1: Geoparsing text column ")
        df = self.ner.geoparse_to_dataframe(df, text_column=text_column)

        print("\n Step 2: Feature selection + model training ")
        self.selector.fit(df)

        print("\n Step 3: Predicting disruptions ")
        df['predicted'] = self.selector.predict_batch(df)

        print("\n✓ Pipeline complete.")
        return df

class LanguageAnalyzer:
    """Dedicated language analysis."""

    def __init__(self, output_dir: str = 'visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def analyze_distribution(self, df: pd.DataFrame,
                              lang_col: str = 'language_name',
                              save_plots: bool = True) -> pd.DataFrame:
        if lang_col not in df.columns:
            print(f"Column '{lang_col}' not found")
            return pd.DataFrame()

        print("\n" + "=" * 60)
        print("LANGUAGE DISTRIBUTION ANALYSIS")
        print("=" * 60)

        lang_counts = df[lang_col].value_counts()
        total       = len(df)
        print(f"\nTotal records: {total:,}")
        print(f"\nLanguage Distribution:")
        for lang, count in lang_counts.items():
            print(f"  {lang}: {count:,} ({count/total*100:.2f}%)")

        summary_df = lang_counts.reset_index()
        summary_df.columns = ['language', 'count']
        summary_df['percentage'] = (summary_df['count'] / total * 100).round(2)

        if save_plots:
            self.plot_language_distribution(df, lang_col)

        return summary_df

    def plot_language_distribution(self, df: pd.DataFrame,
                                    lang_col: str = 'language_name'):
        if lang_col not in df.columns:
            print(f"Column '{lang_col}' not found")
            return

        lang_counts = df[lang_col].value_counts().head(10)
        if lang_counts.empty:
            print("  No language data to plot")
            return

        n   = len(lang_counts)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Language Distribution', fontsize=14, fontweight='bold')

        axes[0].barh(range(n), lang_counts.values,
                     color=plt.cm.viridis(np.linspace(0, 1, n)))
        axes[0].set_yticks(range(n))
        axes[0].set_yticklabels(lang_counts.index)
        axes[0].set_title('Top 10 Languages (count)')
        axes[0].set_xlabel('Number of Alerts')
        axes[0].grid(True, alpha=0.3, axis='x')

        axes[1].pie(lang_counts.values, labels=lang_counts.index,
                    autopct='%1.1f%%',
                    colors=plt.cm.Pastel1(np.linspace(0, 1, n)),
                    startangle=90)
        axes[1].set_title('Language Share (Top 10)')

        plt.tight_layout()

        os.makedirs(self.output_dir, exist_ok=True)
        path = f'{self.output_dir}/language_distribution.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close('all')
        print(f"✓ Saved → {path}")

    def calculate_confidence_stats(self, df: pd.DataFrame,
                                    confidence_col: str = 'confidence') -> Dict:
        if confidence_col not in df.columns:
            print(f"Column '{confidence_col}' not found")
            return {}

        conf  = df[confidence_col].dropna()
        total = len(conf)
        stats = {
            'mean':                float(conf.mean()),
            'median':              float(conf.median()),
            'std':                 float(conf.std()),
            'low_confidence':      int((conf < 0.9).sum()),
            'high_confidence':     int((conf > 0.95).sum()),
            'low_confidence_pct':  float((conf < 0.9).sum() / total * 100),
            'high_confidence_pct': float((conf > 0.95).sum() / total * 100),
        }

        print(f"\nConfidence Statistics:")
        for k, v in stats.items():
            fmt = (f'{v:.2f}%' if 'pct' in k else
                   f'{v:.4f}' if isinstance(v, float) else f'{v:,}')
            print(f"  {k}: {fmt}")
        return stats

    def identify_unknown_languages(self, df: pd.DataFrame,
                                    lang_col: str = 'language_name') -> pd.DataFrame:
        if lang_col not in df.columns:
            return pd.DataFrame()
        mask   = df[lang_col].isin(['unknown', 'UNKNOWN']) | df[lang_col].isna()
        unk_df = df[mask].copy()
        print(f"\nUnknown-language alerts: {len(unk_df):,} "
              f"({len(unk_df)/len(df)*100:.2f}%)")
        if 'text_length' in unk_df.columns and not unk_df.empty:
            print(f"  Avg text length: {unk_df['text_length'].mean():.1f} chars")
        return unk_df

    def export_summary(self, df: pd.DataFrame,
                        filename: str = 'language_summary.csv',
                        lang_col: str = 'language_name') -> str:
        summary = self.analyze_distribution(df, lang_col, save_plots=False)
        if summary.empty:
            return ''
        path = os.path.join(self.output_dir, filename)
        summary.to_csv(path, index=False)
        print(f"✓ Language summary saved → {path}")
        return path

class GTFSAnalysisPipeline:
    """Complete GTFS-RT alerts analysis pipeline (6-step)."""

    def __init__(self, repo_path=None):
        self.repo_path         = repo_path
        self.data_loader       = GTFSDataLoader(repo_path)
        self.preprocessor      = AlertsPreprocessor()
        self.text_preprocessor = TextPreprocessor()
        self.language_detector = LanguageDetector()
        self.analyzer          = AlertsAnalyzer()
        self.exporter          = ResultsExporter()
        self.filter            = AlertsFilter()
        self.parsed_feeds      = None
        self._ner              = None
        self._sentiment        = None
        self._topic_model      = None

    @property
    def ner(self):
        if self._ner is None:
            self._ner = MultilingualNER()
        return self._ner

    @property
    def sentiment(self):
        if self._sentiment is None:
            self._sentiment = MultilingualSentiment()
        return self._sentiment

    @property
    def topic_model(self):
        if self._topic_model is None:
            self._topic_model = MultilingualTopicModeling()
        return self._topic_model

    def run_complete_analysis(self):
        print("\n" + "=" * 80)
        print(" GTFS-RT ALERTS ANALYSIS PIPELINE")
        print("=" * 80)

        try:
            print("\n[1/6] Loading data...")
            with Timer():
                alerts_df, parsed_feeds = self.data_loader.load_gtfs_data()
                self.parsed_feeds = parsed_feeds

            if alerts_df is None or alerts_df.empty:
                print(" No data loaded")
                return None

            print("\n[2/6] Preprocessing...")
            with Timer():
                processed_df = self.preprocessor.preprocess_alerts_data(alerts_df)

            print("\n[3/6] Advanced text preprocessing...")
            with Timer():
                processed_df = self.text_preprocessor.preprocess_dataframe(processed_df)

            print("\n[4/6] Detecting languages...")
            if 'combined_text' in processed_df.columns:
                with Timer():
                    lang_df = self.language_detector.detect_batch(
                        processed_df['combined_text'].tolist()
                    )
                    processed_df = pd.concat(
                        [processed_df.reset_index(drop=True), lang_df], axis=1
                    )

            print("\n[5/6] Running analysis...")
            with Timer():
                self.analyzer.analyze_basic(processed_df)
                self.analyzer.analyze_temporal_patterns(processed_df)

            print("\n[6/6] Exporting results...")
            with Timer():
                self.exporter.export_analysis_results(processed_df)

            print("\n" + "=" * 80)
            print(" PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print(f"Final DataFrame: {len(processed_df):,} records")
            return processed_df

        except Exception as e:
            print(f"\n Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return None


# SECTION 13: EVENT RECONSTRUCTOR

class GTFSEventReconstructor:
    """Converts high-frequency GTFS-RT snapshots into event-level alerts."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare()

    def _prepare(self):
        self.df['feed_timestamp'] = pd.to_datetime(
            self.df['feed_timestamp'], errors='coerce')
        if 'active_period_start' in self.df.columns:
            self.df['active_period_start'] = pd.to_datetime(
                self.df['active_period_start'], errors='coerce')
        if 'active_period_end' in self.df.columns:
            self.df['active_period_end'] = pd.to_datetime(
                self.df['active_period_end'], errors='coerce')
        self.df = self.df.sort_values('feed_timestamp')

    def reconstruct_events(self) -> pd.DataFrame:
        """Aggregate snapshot rows into one row per unique alert event.

        ADDITION: Column existence guard — NLP columns (language_code,
        language_name, etc.) may not be present if the NLP pipeline has
        not been run yet.
        """
        base_agg = {
            'feed_timestamp':    ['min', 'max'],
            'active_period_start': 'min',
            'active_period_end':   'max',
            'alert_id':          'last',
            'cause_id':          'last',
            'cause':             'last',
            'effect_id':         'last',
            'effect':            'last',
            'agency_id':         'last',
            'route_id':          'last',
            'route_type':        'last',
            'stop_id':           'last',
            'description_text':  'last',
        }

        # only include NLP columns if they actually exist
        optional_cols = ['combined_text', 'language_code', 'language_name', 'confidence']
        for col in optional_cols:
            if col in self.df.columns:
                base_agg[col] = 'last'

        grouped   = self.df.groupby('rt_id')
        event_df  = grouped.agg(base_agg)
        event_df.columns = ['_'.join(filter(None, c)) if isinstance(c, tuple) else c
                             for c in event_df.columns]

        # Flatten multi-level column names from feed_timestamp min/max
        rename_map = {}
        for col in event_df.columns:
            if 'feed_timestamp_min' in col:
                rename_map[col] = 'feed_timestamp_min'
            elif 'feed_timestamp_max' in col:
                rename_map[col] = 'feed_timestamp_max'
        event_df = event_df.rename(columns=rename_map).reset_index()
        return event_df

    def compute_lifecycle(self, event_df: pd.DataFrame) -> pd.DataFrame:
        """Compute duration metrics from active period timestamps.

        ADDITION: A 'period_source' column flags whether start/end timestamps
        were declared in the feed or inferred from snapshot boundaries,
        preventing silent substitution of semantically different values.
        """
        event_df = event_df.copy()

        # Track whether start/end were inferred
        event_df['start_inferred'] = event_df['active_period_start'].isna()
        event_df['end_inferred']   = event_df['active_period_end'].isna()

        event_df['active_period_start'] = event_df['active_period_start'].fillna(
            event_df['feed_timestamp_min'])
        event_df['active_period_end']   = event_df['active_period_end'].fillna(
            event_df['feed_timestamp_max'])

        duration_seconds = (
            event_df['active_period_end'] - event_df['active_period_start']
        ).dt.total_seconds()

        event_df['active_period_duration_seconds'] = duration_seconds
        event_df['active_period_duration_days']    = duration_seconds / (60 * 60 * 24)
        event_df['alert_duration_min']             = duration_seconds / 60

        return event_df

    def clean_durations(self, event_df: pd.DataFrame, max_days: int = 14) -> pd.DataFrame:
        """Nullify unrealistic durations.

        ADDITION: max_days is now a configurable parameter (default 14)
        rather than a hard-coded constant, as the threshold is domain-dependent.
        """
        max_seconds = max_days * 24 * 60 * 60
        dur_cols    = ['active_period_duration_seconds',
                       'active_period_duration_days',
                       'alert_duration_min']

        event_df.loc[event_df['active_period_duration_seconds'] > max_seconds, dur_cols] = np.nan
        event_df.loc[event_df['active_period_duration_seconds'] < 0,           dur_cols] = np.nan

        return event_df

    def build_modeling_dataset(self) -> pd.DataFrame:
        """Build the final clean temporal modeling dataset.

        ADDITION: Added month and is_weekend features alongside
        hour and day_of_week.
        REMOVAL: Cyclical encoding removed from here — it is now applied
        universally in AlertsPreprocessor._add_cyclical_time_features().
        """
        event_df = self.reconstruct_events()
        event_df = self.compute_lifecycle(event_df)
        event_df = self.clean_durations(event_df)

        event_df['hour']        = event_df['active_period_start'].dt.hour
        event_df['day_of_week'] = event_df['active_period_start'].dt.dayofweek  # int 0-6
        event_df['day_name']    = event_df['active_period_start'].dt.day_name() # str label
        event_df['date']        = event_df['active_period_start'].dt.date
        event_df['month']       = event_df['active_period_start'].dt.month
        event_df['is_weekend']  = event_df['day_of_week'].isin([5, 6]).astype(int)

        return event_df

"""
hybrid_pipeline_classes.py
══════════════════════════════════════════════════════════════════════════════
Hybrid Clustering → Supervised Classification Pipeline
Fully object-oriented implementation for transit disruption alerts.

Class hierarchy
─
  FeatureExtractor          – embeddings + structured features + PCA
  AlertClusterer            – HDBSCAN clustering + cluster inspection
  LabelMapper               – maps cluster IDs → human-readable categories
  DataSplitter              – stratified train/val/test split + SMOTE
  ModelTrainer              – trains RF · XGBoost · MLP with CV
  HyperparameterTuner       – RandomizedSearchCV wrapper
  ModelEvaluator            – metrics, plots, threshold analysis
  HybridPipeline            – orchestrates all classes end-to-end

Usage (end of file)
─
  pipeline = HybridPipeline(df=location_disruption_df)
  pipeline.run()
"""

#  Standard library ─
import gc
import warnings
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("hybrid_pipeline")

#  Numeric / DataFrame ─
import numpy as np
import pandas as pd

#  Visualisation ─
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#  Sklearn ─
from sklearn.decomposition        import PCA
from sklearn.ensemble             import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics              import (
    ConfusionMatrixDisplay, average_precision_score,
    classification_report, confusion_matrix,
    f1_score, precision_recall_curve,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection      import (
    RandomizedSearchCV, StratifiedKFold,
    cross_val_score, train_test_split,
)
from sklearn.neural_network       import MLPClassifier
from sklearn.preprocessing        import StandardScaler, label_binarize
from sklearn.utils.class_weight   import compute_class_weight

#  Imbalanced-learn ─
from imblearn.over_sampling import SMOTE

#  XGBoost ─
import xgboost as xgb

#  Clustering
import hdbscan

#  Embeddings
from sentence_transformers import SentenceTransformer

#  Serialisation ─
import joblib
from scipy.stats import randint, uniform

# 0.  CONFIG DATACLASS

@dataclass
class PipelineConfig:
    """
    Central configuration for the entire hybrid pipeline.

    All tunable parameters live here so nothing is hard-coded
    inside the individual class methods.

    Definition

    A frozen configuration bag passed to every class at construction
    time. Changing one value here propagates consistently through
    the whole pipeline.

    Parameters

    random_state      : Seed for reproducibility across sklearn, numpy, XGBoost.
    sample_size       : Max rows used to *fit* HDBSCAN (keeps RAM manageable).
    embed_model       : HuggingFace / SBERT model name for sentence embeddings.
    pca_variance      : Fraction of variance retained after PCA compression.
    min_cluster_size  : HDBSCAN minimum cluster size.
    min_samples       : HDBSCAN core-point density threshold.
    test_size         : Fraction of labeled data reserved for final test set.
    val_size          : Fraction of *remaining* data reserved for validation.
    cv_folds          : Number of stratified cross-validation folds.
    tuning_iter       : RandomizedSearchCV iterations for hyperparameter search.
    tolerance         : Time-window tolerance (minutes) for alert overlap joins.
    output_dir        : Directory where models and plots are saved.
    label_col         : Name of the target column created by LabelMapper.
    """
    random_state    : int   = 42
    sample_size     : int   = 50_000
    embed_model     : str   = "paraphrase-multilingual-MiniLM-L12-v2"
    pca_variance    : float = 0.95
    min_cluster_size: int   = 80
    min_samples     : int   = 10
    test_size       : float = 0.15
    val_size        : float = 0.15
    cv_folds        : int   = 5
    tuning_iter     : int   = 40
    tolerance_mins  : int   = 90
    output_dir      : Path  = Path("models")
    label_col       : str   = "alert_category"


# 1.  FEATURE EXTRACTOR

class FeatureExtractor:
    """
    Definition

    Converts raw alert text and structured columns into a single
    numeric feature matrix ready for clustering and classification.

    The matrix combines:
      • Dense sentence embeddings  (multilingual SBERT)
      • Scaled structured features (urgency, sentiment, NER counts, …)
      • One-hot encoded categoricals (cause, effect)

    PCA is applied as a final compression step so that HDBSCAN
    operates in a manageable number of dimensions.

    Step usage

    Called by HybridPipeline.run() as Step 1.

        extractor = FeatureExtractor(config, df)
        X_full, X_pca = extractor.extract()

    Public attributes (set after extract())

    embeddings    : np.ndarray  shape (n, embed_dim)
    struct_arr    : np.ndarray  shape (n, n_struct)
    X_full        : np.ndarray  shape (n, embed_dim + n_struct)
    X_pca         : np.ndarray  shape (n, n_pca_components)
    numeric_cols  : List[str]   structured feature names
    pca           : fitted sklearn PCA object
    scaler        : fitted StandardScaler for structured features
    """

    STRUCTURED_CANDIDATES = [
        "urgency_score", "score", "sentiment",
        "duration_weighted_alert_influence",
        "spatial_extent", "affected_passengers",
        "word_count", "has_time_ref", "has_location_ref",
        "active_period_duration_hours",
    ]
    CATEGORICAL_COLS = ["cause", "effect"]
    LIST_COLS        = ["locations", "times", "alert_type_tags"]

    def __init__(self, config: PipelineConfig, df: pd.DataFrame):
        self.cfg = config
        self.df  = df.copy().reset_index(drop=True)

        # Resolved at runtime
        self.text_col    : Optional[str]  = None
        self.numeric_cols: List[str]      = []
        self.embeddings  : np.ndarray     = np.empty(0)
        self.struct_arr  : np.ndarray     = np.empty(0)
        self.X_full      : np.ndarray     = np.empty(0)
        self.X_pca       : np.ndarray     = np.empty(0)
        self.pca         : Optional[PCA]  = None
        self.scaler      : StandardScaler = StandardScaler()

    #  public

    def extract(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full extraction pipeline.

        Returns
        ─
        X_full : combined embeddings + structured features (pre-PCA)
        X_pca  : PCA-compressed version used for clustering
        """
        self._resolve_text_col()
        self._build_derived_features()
        self._embed_text()
        self._build_structured()
        self._combine()
        self._compress_pca()
        return self.X_full, self.X_pca

    #  private ─

    def _resolve_text_col(self):
        """Pick the first available free-text column."""
        for candidate in ["description_text", "header_text", "combined_text"]:
            if candidate in self.df.columns:
                self.text_col = candidate
                break
        log.info("Text column: %s", self.text_col)

    def _build_derived_features(self):
        """
        Create derived numeric columns that don't exist yet:
          • active_period_duration_hours  – from start/end timestamps
          • n_<list_col>                  – count of items in list columns
          • one-hot dummies for cause/effect
        """
        # Duration
        if ("active_period_start" in self.df.columns and
                "active_period_end" in self.df.columns):
            self.df["active_period_duration_hours"] = (
                (self.df["active_period_end"] - self.df["active_period_start"])
                .dt.total_seconds().div(3600).clip(0, 720)
            )

        # NER / list counts
        for col in self.LIST_COLS:
            if col in self.df.columns:
                self.df[f"n_{col}"] = self.df[col].apply(
                    lambda x: len(x) if isinstance(x, list) else 0
                )

        # Categorical dummies
        for cat in self.CATEGORICAL_COLS:
            if cat in self.df.columns:
                dummies = pd.get_dummies(self.df[cat].astype(str), prefix=cat)
                self.df = pd.concat([self.df, dummies], axis=1)

    def _embed_text(self):
        """Encode alert text with a multilingual sentence transformer."""
        texts = (
            self.df[self.text_col].fillna("").astype(str).tolist()
            if self.text_col else [""] * len(self.df)
        )
        log.info("Encoding %d texts with %s …", len(texts), self.cfg.embed_model)
        encoder        = SentenceTransformer(self.cfg.embed_model)
        self.embeddings = encoder.encode(
            texts, batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        log.info("Embeddings shape: %s", self.embeddings.shape)

    def _build_structured(self):
        """Scale structured numeric features into a numpy array."""
        base_cols = [c for c in self.STRUCTURED_CANDIDATES if c in self.df.columns]
        ner_cols  = [f"n_{c}" for c in self.LIST_COLS if f"n_{c}" in self.df.columns]
        dummy_cols = [c for c in self.df.columns
                      if any(c.startswith(f"{cat}_") for cat in self.CATEGORICAL_COLS)]

        self.numeric_cols = base_cols + ner_cols + dummy_cols
        if not self.numeric_cols:
            self.struct_arr = np.empty((len(self.df), 0))
            return

        raw = self.df[self.numeric_cols].fillna(
            self.df[self.numeric_cols].median(numeric_only=True)
        )
        self.struct_arr = self.scaler.fit_transform(raw)
        log.info("Structured features: %d", self.struct_arr.shape[1])

    def _combine(self):
        """Horizontally stack embeddings and structured arrays."""
        parts = [self.embeddings]
        if self.struct_arr.shape[1] > 0:
            parts.append(self.struct_arr)
        self.X_full = np.hstack(parts).astype(np.float32)
        log.info("Combined matrix: %s", self.X_full.shape)

    def _compress_pca(self):
        """Reduce dimensionality to cfg.pca_variance explained variance."""
        self.pca   = PCA(n_components=self.cfg.pca_variance,
                         random_state=self.cfg.random_state)
        self.X_pca = self.pca.fit_transform(self.X_full)
        log.info("PCA → %d dims (%s variance)",
                 self.X_pca.shape[1], f"{self.cfg.pca_variance:.0%}")

# 2.  ALERT CLUSTERER


class AlertClusterer:
    """
    Definition

    Applies HDBSCAN density-based clustering to discover natural
    groupings in the alert feature space — without requiring a
    pre-specified number of clusters.

    HDBSCAN is preferred over k-means here because:
      • It handles arbitrary cluster shapes.
      • It explicitly marks low-density points as noise (-1).
      • It returns soft-membership probabilities (cluster_strength).

    Fitting on a sample (cfg.sample_size) and predicting on the
    full dataset via approximate_predict() keeps memory usage bounded
    for 176k+ row datasets.

    Step usage

    Called by HybridPipeline.run() as Step 2.

        clusterer = AlertClusterer(config)
        df = clusterer.fit_predict(df, X_pca)

    Public attributes (set after fit_predict())

    clusterer    : fitted hdbscan.HDBSCAN object
    n_clusters   : number of clusters found (excluding noise)
    noise_pct    : fraction of rows assigned to noise class
    """

    def __init__(self, config: PipelineConfig):
        self.cfg        = config
        self.clusterer  = None
        self.n_clusters = 0
        self.noise_pct  = 0.0

    #  public

    def fit_predict(self, df: pd.DataFrame, X_pca: np.ndarray) -> pd.DataFrame:
        """
        Fit HDBSCAN on a random sample, then soft-assign all rows.

        Parameters

        df    : source DataFrame (cluster_id and cluster_strength columns added)
        X_pca : PCA-compressed feature matrix aligned to df's index

        Returns
        ─
        df with two new columns: cluster_id, cluster_strength
        """
        df = df.copy()
        rng = np.random.default_rng(self.cfg.random_state)
        sample_idx = rng.choice(
            len(X_pca),
            size=min(self.cfg.sample_size, len(X_pca)),
            replace=False,
        )

        log.info("Fitting HDBSCAN on %d rows …", len(sample_idx))
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.cfg.min_cluster_size,
            min_samples=self.cfg.min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        self.clusterer.fit(X_pca[sample_idx])

        labels, strengths      = hdbscan.approximate_predict(self.clusterer, X_pca)
        df["cluster_id"]       = labels
        df["cluster_strength"] = strengths

        self.n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.noise_pct  = (labels == -1).mean() * 100

        log.info("Clusters: %d  |  Noise: %.1f%%", self.n_clusters, self.noise_pct)
        return df

    def inspect(self, df: pd.DataFrame, text_col: Optional[str],
                top_n_keywords: int = 10):
        """
        Print a diagnostic report for each cluster:
          • Size
          • Top TF-IDF bigrams
          • Most common entity types (locations, alert tags)
          • Three sample alert texts

        Step usage

        Called interactively after fit_predict() so the analyst
        can decide human-readable labels before supervised training.

            clusterer.inspect(df, text_col="description_text")
        """
        for cid in sorted(df["cluster_id"].unique()):
            if cid == -1:
                continue
            mask   = df["cluster_id"] == cid
            subset = df[mask]
            print(f"\n  ┌ Cluster {cid}  ({mask.sum():,} rows) " + "─" * 40)

            # TF-IDF keywords
            if text_col and text_col in df.columns:
                texts_c = subset[text_col].dropna().astype(str).tolist()
                if len(texts_c) >= 3:
                    tfidf = TfidfVectorizer(max_features=500,
                                            stop_words="english",
                                            ngram_range=(1, 2))
                    try:
                        scores = np.asarray(
                            tfidf.fit_transform(texts_c).mean(axis=0)
                        ).flatten()
                        kws = [tfidf.get_feature_names_out()[i]
                               for i in scores.argsort()[::-1][:top_n_keywords]]
                        print(f"  │  Keywords : {', '.join(kws)}")
                    except Exception:
                        pass
                    print("  │  Samples  :")
                    for t in texts_c[:3]:
                        print(f"  │    · {t[:110]}")

            # Entity counts
            for col in ["locations", "alert_type_tags"]:
                if col in subset.columns:
                    top = subset[col].explode().value_counts().head(5)
                    print(f"  │  Top {col}: {top.index.tolist()}")
            print("  └" + "─" * 55)

# 3.  LABEL MAPPER


class LabelMapper:
    """
    Definition

    Translates integer HDBSCAN cluster IDs into human-readable
    alert category strings that become the supervised target labels.

    This is the bridge between the unsupervised and supervised
    halves of the pipeline.  A default mapping is provided; the
    analyst should override it after reading the cluster inspection
    output from AlertClusterer.inspect().

    Step usage

    Called by HybridPipeline.run() as Step 3.

        mapper = LabelMapper(config, cluster_label_map={0: "signal_failure", …})
        df     = mapper.apply(df)

    Noise rows (cluster_id == -1) are always mapped to "noise"
    and excluded from supervised training by DataSplitter.

    Public attributes
    ─
    label_map    : Dict[int, str]  the final {cluster_id: label} mapping
    label_counts : pd.Series       distribution of assigned labels
    """

    DEFAULT_MAP: Dict[int, str] = {
        0:  "signal_failure",
        1:  "planned_maintenance",
        2:  "weather_impact",
        3:  "power_outage",
        4:  "security_incident",
        5:  "overcrowding",
        6:  "equipment_failure",
        7:  "detour",
        8:  "cancellation",
        -1: "noise",
    }

    def __init__(self, config: PipelineConfig,
                 cluster_label_map: Optional[Dict[int, str]] = None):
        self.cfg       = config
        self.label_map = cluster_label_map or self.DEFAULT_MAP.copy()

    #  public

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cfg.label_col column to df based on cluster_id → label mapping.
        Any cluster ID not in label_map gets an auto-generated label.

        Returns
        ─
        df with the new label column added in-place.
        """
        df = df.copy()

        # Auto-fill any unmapped cluster IDs
        for cid in df["cluster_id"].unique():
            if cid not in self.label_map:
                self.label_map[cid] = f"cluster_{cid}"

        df[self.cfg.label_col] = df["cluster_id"].map(self.label_map)
        self.label_counts      = df[self.cfg.label_col].value_counts()

        log.info("Label distribution:\n%s", self.label_counts.to_string())
        return df

    def update(self, cluster_id: int, label: str):
        """
        Interactively update a single cluster's label.

        Step usage

        Called manually in a notebook after inspecting clusters:

            mapper.update(3, "infrastructure_failure")
            df = mapper.apply(df)  # re-apply updated map
        """
        self.label_map[cluster_id] = label
        log.info("Cluster %d → '%s'", cluster_id, label)

# 4.  DATA SPLITTER

class DataSplitter:
    """
    Definition

    Performs stratified train / validation / test splitting and
    applies SMOTE oversampling to the training partition only.

    Stratification ensures each split mirrors the full class
    distribution; SMOTE synthesises minority-class examples so
    that classifiers don't simply learn to predict the majority class.

    SMOTE is applied *only to training data* — never to val or test —
    to avoid inflating evaluation metrics.

    Step usage

    Called by HybridPipeline.run() as Step 4.

        splitter = DataSplitter(config)
        splits   = splitter.split(df_labeled, X_full)

    Public attributes (set after split())

    X_train, y_train : resampled training arrays (post-SMOTE)
    X_val,   y_val   : validation arrays
    X_test,  y_test  : held-out test arrays (never touched until Step 8)
    classes          : sorted unique class labels
    class_weights    : dict mapping class → balanced weight
    scaler           : fitted StandardScaler (applied before MLP)
    """

    def __init__(self, config: PipelineConfig):
        self.cfg          = config
        self.X_train      = self.X_val   = self.X_test   = np.empty(0)
        self.y_train      = self.y_val   = self.y_test   = np.empty(0)
        self.X_train_sc   = self.X_val_sc = self.X_test_sc = np.empty(0)
        self.classes      : np.ndarray   = np.empty(0)
        self.class_weights: Dict         = {}
        self.scaler       : StandardScaler = StandardScaler()

    #  public

    def split(self, df_labeled: pd.DataFrame,
              X_full: np.ndarray) -> "DataSplitter":
        """
        Execute the full split + SMOTE pipeline.

        Parameters

        df_labeled : rows with label_col populated (noise excluded)
        X_full     : full feature matrix aligned to df_labeled's original index

        Returns self (fluent interface).
        """
        labeled_idx = df_labeled.index
        X           = X_full[labeled_idx]
        y           = df_labeled[self.cfg.label_col].values

        # Train+val / test
        X_tv, self.X_test, y_tv, self.y_test = train_test_split(
            X, y,
            test_size=self.cfg.test_size,
            stratify=y,
            random_state=self.cfg.random_state,
        )

        # Train / val
        val_frac = self.cfg.val_size / (1 - self.cfg.test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_tv, y_tv,
            test_size=val_frac,
            stratify=y_tv,
            random_state=self.cfg.random_state,
        )

        log.info("Split  →  train:%d  val:%d  test:%d",
                 len(self.X_train), len(self.X_val), len(self.X_test))

        # Class weights
        self.classes = np.unique(self.y_train)
        weights      = compute_class_weight("balanced",
                                             classes=self.classes,
                                             y=self.y_train)
        self.class_weights = dict(zip(self.classes, weights))

        # SMOTE
        self._apply_smote()

        # Scaled versions for MLP
        self.X_train_sc = self.scaler.fit_transform(self.X_train)
        self.X_val_sc   = self.scaler.transform(self.X_val)
        self.X_test_sc  = self.scaler.transform(self.X_test)

        return self

    #  private ─

    def _apply_smote(self):
        """Oversample minority classes on the training partition only."""
        try:
            sm = SMOTE(random_state=self.cfg.random_state, k_neighbors=5)
            self.X_train, self.y_train = sm.fit_resample(self.X_train, self.y_train)
            log.info("SMOTE → train size: %d", len(self.X_train))
        except Exception as exc:
            log.warning("SMOTE skipped: %s", exc)

# 5.  MODEL TRAINER

class ModelTrainer:
    """
    Definition

    Instantiates, trains, and cross-validates the three classifiers:

      • Random Forest  – strong baseline; interpretable feature importances.
      • XGBoost        – gradient boosting; often best on tabular+embedding data.
      • MLP            – small neural network; benefits from scaled inputs.

    Each model is trained on the SMOTE-resampled training set.
    Stratified k-fold cross-validation provides an unbiased F1 estimate
    before any test data is touched.

    Step usage

    Called by HybridPipeline.run() as Step 5.

        trainer = ModelTrainer(config, splits)
        trainer.train_all()

    Public attributes (set after train_all())

    models      : Dict[str, sklearn estimator]  trained model objects
    cv_results  : Dict[str, dict]  mean/std CV F1 per model name
    val_results : Dict[str, float] validation F1 per model name
    """

    def __init__(self, config: PipelineConfig, splits: DataSplitter):
        self.cfg    = config
        self.splits = splits
        self.models : Dict[str, Any]   = {}
        self.cv_results : Dict         = {}
        self.val_results: Dict         = {}

    #  public

    def build_models(self) -> Dict[str, Any]:
        """
        Construct (but do not yet train) the three estimators.

        Override this method to swap in different architectures
        without touching any other class.
        """
        return {
            "Random Forest": RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=self.cfg.random_state,
            ),
            "XGBoost": xgb.XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                n_jobs=-1,
                random_state=self.cfg.random_state,
            ),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=self.cfg.random_state,
            ),
        }

    def train_all(self):
        """
        Train every model, run CV, compute validation F1.

        MLP receives StandardScaler-transformed inputs;
        tree-based models receive the raw (SMOTE) training data.
        """
        self.models = self.build_models()
        cv = StratifiedKFold(
            n_splits=self.cfg.cv_folds, shuffle=True,
            random_state=self.cfg.random_state,
        )
        s = self.splits

        for name, model in self.models.items():
            log.info("Training %s …", name)
            X_fit = s.X_train_sc if name == "MLP" else s.X_train
            X_cv  = s.X_train_sc if name == "MLP" else s.X_train
            X_ev  = s.X_val_sc   if name == "MLP" else s.X_val

            model.fit(X_fit, s.y_train)

            scores = cross_val_score(model, X_cv, s.y_train,
                                     cv=cv, scoring="f1_weighted", n_jobs=-1)
            val_f1 = f1_score(s.y_val, model.predict(X_ev), average="weighted")

            self.cv_results[name]  = {"mean": scores.mean(), "std": scores.std()}
            self.val_results[name] = val_f1

            log.info("  CV F1: %.4f ± %.4f  |  Val F1: %.4f",
                     scores.mean(), scores.std(), val_f1)

# 6.  HYPERPARAMETER TUNER

class HyperparameterTuner:
    """
    Definition

    Runs RandomizedSearchCV on the best-performing model from
    ModelTrainer to find improved hyperparameters without exhaustive
    grid search.

    Randomised search is preferred over GridSearchCV because:
      • Each iteration samples the full parameter space rather than
        testing every combination.
      • A fixed budget (cfg.tuning_iter) controls compute cost.
      • It often finds near-optimal parameters in far fewer evaluations.

    Step usage

    Called by HybridPipeline.run() as Step 6.

        tuner = HyperparameterTuner(config, trainer, splits)
        tuner.tune()

    Public attributes (set after tune())

    best_model_name : str   name of the model that was tuned
    best_params     : dict  winning hyperparameter values
    tuned_model     : fitted sklearn estimator with best params
    """

    PARAM_GRIDS: Dict[str, Dict] = {
        "Random Forest": {
            "n_estimators":     randint(200, 600),
            "max_depth":        [None, 10, 20, 30],
            "min_samples_leaf": randint(1, 10),
            "max_features":     ["sqrt", "log2", 0.5],
        },
        "XGBoost": {
            "n_estimators":     randint(200, 600),
            "max_depth":        randint(3, 10),
            "learning_rate":    uniform(0.01, 0.2),
            "subsample":        uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_alpha":        uniform(0, 1),
            "reg_lambda":       uniform(0.5, 2),
        },
        "MLP": {
            "hidden_layer_sizes": [(128, 64), (256, 128, 64), (512, 256, 128)],
            "alpha":              uniform(1e-5, 1e-2),
            "learning_rate_init": uniform(1e-4, 1e-2),
        },
    }

    def __init__(self, config: PipelineConfig,
                 trainer: ModelTrainer, splits: DataSplitter):
        self.cfg             = config
        self.trainer         = trainer
        self.splits          = splits
        self.best_model_name : str = ""
        self.best_params     : Dict = {}
        self.tuned_model     : Any  = None

    #  public

    def tune(self):
        """
        Identify the best model by validation F1, then run
        RandomizedSearchCV to improve its hyperparameters.
        Updates trainer.models[best_model_name] with the tuned estimator.
        """
        self.best_model_name = max(
            self.trainer.val_results,
            key=lambda k: self.trainer.val_results[k],
        )
        log.info("Tuning %s …", self.best_model_name)

        s       = self.splits
        X_tune  = (s.X_train_sc if self.best_model_name == "MLP"
                   else s.X_train)
        base_model = self.trainer.models[self.best_model_name]

        search = RandomizedSearchCV(
            base_model,
            param_distributions=self.PARAM_GRIDS[self.best_model_name],
            n_iter=self.cfg.tuning_iter,
            scoring="f1_weighted",
            cv=StratifiedKFold(3, shuffle=True,
                               random_state=self.cfg.random_state),
            n_jobs=-1,
            random_state=self.cfg.random_state,
            verbose=1,
        )
        search.fit(X_tune, s.y_train)

        self.tuned_model  = search.best_estimator_
        self.best_params  = search.best_params_
        self.trainer.models[self.best_model_name] = self.tuned_model

        X_ev  = (s.X_val_sc if self.best_model_name == "MLP" else s.X_val)
        val_f1_tuned = f1_score(s.y_val, self.tuned_model.predict(X_ev),
                                 average="weighted")
        log.info("Val F1 after tuning: %.4f  |  Params: %s",
                 val_f1_tuned, self.best_params)

# geoparse_to_dataframe
# 7.  MODEL EVALUATOR

class ModelEvaluator:
    """
    Definition

    Runs all three trained (and one tuned) models against the
    held-out test set and produces:
      • Per-class classification reports
      • Weighted ROC-AUC scores
      • 12-panel evaluation figure:
          - CV F1 / Test F1 / ROC-AUC bar charts
          - Normalised confusion matrices
          - Per-class ROC curves (best model)
          - Per-class Precision-Recall curves (best model)
          - Threshold sweep → optimal decision boundary
          - Aggregated feature importance chart

    The threshold analysis answers: "at what confidence cut-off does
    macro F1 peak?" — useful for production systems where you want
    to abstain rather than give low-confidence predictions.

    Step usage

    Called by HybridPipeline.run() as Step 7.

        evaluator = ModelEvaluator(config, trainer, splits, extractor)
        evaluator.evaluate()
        evaluator.plot()
        evaluator.print_summary()
    """

    COLORS = {"Random Forest": "#2563EB", "XGBoost": "#16A34A", "MLP": "#DC2626"}

    def __init__(self, config: PipelineConfig,
                 trainer: ModelTrainer,
                 splits: DataSplitter,
                 extractor: FeatureExtractor):
        self.cfg       = config
        self.trainer   = trainer
        self.splits    = splits
        self.extractor = extractor

        self.test_results  : Dict = {}
        self.best_model_name: str = ""
        self.optimal_threshold: float = 0.5

    #  public

    def evaluate(self):
        """
        Run every model on the test set and collect metrics.
        Populates self.test_results.
        """
        s = self.splits
        for name, model in self.trainer.models.items():
            X_ev   = s.X_test_sc if name == "MLP" else s.X_test
            y_pred = model.predict(X_ev)
            y_prob = (model.predict_proba(X_ev)
                      if hasattr(model, "predict_proba") else None)

            roc_auc = (
                roc_auc_score(s.y_test, y_prob,
                              multi_class="ovr", average="weighted")
                if y_prob is not None else np.nan
            )

            self.test_results[name] = {
                "y_pred":  y_pred,
                "y_proba": y_prob,
                "report":  classification_report(s.y_test, y_pred,
                                                  output_dict=True),
                "roc_auc": roc_auc,
            }
            print(f"\n {name} ")
            print(classification_report(s.y_test, y_pred, digits=4))
            print(f"  ROC-AUC (weighted OvR) : {roc_auc:.4f}")

        self.best_model_name = max(
            self.test_results,
            key=lambda n: self.test_results[n]["report"]["weighted avg"]["f1-score"],
        )

    def plot(self):
        """Generate and save the 12-panel evaluation figure."""
        s       = self.splits
        names   = list(self.trainer.models.keys())
        classes = s.classes
        fig     = plt.figure(figsize=(22, 26))
        fig.suptitle(
            "Hybrid Clustering → Classification  ·  Full Evaluation",
            fontsize=15, fontweight="bold", y=0.99,
        )
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.48, wspace=0.38)

        self._plot_cv_f1(fig.add_subplot(gs[0, 0]), names)
        self._plot_test_f1(fig.add_subplot(gs[0, 1]), names)
        self._plot_roc_auc(fig.add_subplot(gs[0, 2]), names)
        for i, name in enumerate(names):
            self._plot_confusion(fig.add_subplot(gs[1, i]), name, classes)
        self._plot_roc_curves(fig.add_subplot(gs[2, :2]), classes)
        self._plot_pr_curves(fig.add_subplot(gs[2, 2]),   classes)
        self._plot_threshold(fig.add_subplot(gs[3, :2]),  classes)
        self._plot_feature_importance(fig.add_subplot(gs[3, 2]))

        out = self.cfg.output_dir / "hybrid_classification_evaluation.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.show()
        log.info("Plot saved → %s", out)

    def print_summary(self):
        """Print a comparison table and announce the winning model."""
        rows = []
        for name in self.test_results:
            r = self.test_results[name]["report"]
            rows.append({
                "Model":          name,
                "CV F1":          f"{self.trainer.cv_results[name]['mean']:.4f}",
                "Val F1":         f"{self.trainer.val_results[name]:.4f}",
                "Test F1":        f"{r['weighted avg']['f1-score']:.4f}",
                "Precision":      f"{r['weighted avg']['precision']:.4f}",
                "Recall":         f"{r['weighted avg']['recall']:.4f}",
                "ROC-AUC":        f"{self.test_results[name]['roc_auc']:.4f}",
            })
        display(pd.DataFrame(rows))
        print(f"\n✔  Best model : {self.best_model_name}")
        print(f"   Optimal threshold : {self.optimal_threshold:.3f}")

    #  private plot helpers

    def _bar(self, ax, names, values, title, ylabel):
        bars = ax.bar(names, values,
                      color=[self.COLORS[n] for n in names], alpha=0.82)
        ax.set_ylim(max(0, min(values) - 0.1), 1.0)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    def _plot_cv_f1(self, ax, names):
        vals = [self.trainer.cv_results[n]["mean"] for n in names]
        errs = [self.trainer.cv_results[n]["std"]  for n in names]
        bars = ax.bar(names, vals, yerr=errs, capsize=6,
                      color=[self.COLORS[n] for n in names], alpha=0.82)
        ax.set_ylim(max(0, min(vals) - 0.1), 1.0)
        ax.set_title("CV F1 (5-fold)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Weighted F1", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    def _plot_test_f1(self, ax, names):
        vals = [self.test_results[n]["report"]["weighted avg"]["f1-score"]
                for n in names]
        self._bar(ax, names, vals, "Test F1 Score", "Weighted F1")

    def _plot_roc_auc(self, ax, names):
        vals = [self.test_results[n]["roc_auc"] for n in names]
        self._bar(ax, names, vals, "ROC-AUC (weighted OvR)", "ROC-AUC")

    def _plot_confusion(self, ax, name, classes):
        y_pred = self.test_results[name]["y_pred"]
        cm     = confusion_matrix(self.splits.y_test, y_pred,
                                  labels=classes, normalize="true")
        ConfusionMatrixDisplay(cm, display_labels=classes).plot(
            ax=ax, colorbar=False, xticks_rotation=45, cmap="Blues"
        )
        ax.set_title(f"{name} – Confusion Matrix", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)

    def _plot_roc_curves(self, ax, classes):
        y_prob = self.test_results[self.best_model_name]["y_proba"]
        if y_prob is None:
            return
        y_bin = label_binarize(self.splits.y_test, classes=classes)
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            auc_i = roc_auc_score(y_bin[:, i], y_prob[:, i])
            ax.plot(fpr, tpr, lw=1.5, alpha=0.8,
                    label=f"{cls} (AUC={auc_i:.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("FPR", fontsize=10); ax.set_ylabel("TPR", fontsize=10)
        ax.set_title(f"ROC Curves – {self.best_model_name}",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, ncol=2, loc="lower right")
        ax.grid(alpha=0.25)

    def _plot_pr_curves(self, ax, classes):
        y_prob = self.test_results[self.best_model_name]["y_proba"]
        if y_prob is None:
            return
        y_bin = label_binarize(self.splits.y_test, classes=classes)
        for i, cls in enumerate(classes):
            p, r, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
            ap = average_precision_score(y_bin[:, i], y_prob[:, i])
            ax.plot(r, p, lw=1.5, alpha=0.8, label=f"{cls} (AP={ap:.2f})")
        ax.set_xlabel("Recall", fontsize=10); ax.set_ylabel("Precision", fontsize=10)
        ax.set_title(f"Precision-Recall – {self.best_model_name}",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.25)

    def _plot_threshold(self, ax, classes):
        y_prob = self.test_results[self.best_model_name]["y_proba"]
        if y_prob is None:
            return
        thresholds = np.linspace(0.1, 0.9, 80)
        macro_f1s  = []
        for thr in thresholds:
            preds = [classes[np.argmax(row)]
                     if row.max() >= thr else None for row in y_prob]
            valid = [(yt, yp) for yt, yp in zip(self.splits.y_test, preds)
                     if yp is not None]
            if len(valid) > 10:
                yt_v, yp_v = zip(*valid)
                macro_f1s.append(
                    f1_score(yt_v, yp_v, average="macro", zero_division=0)
                )
            else:
                macro_f1s.append(np.nan)

        best_idx = int(np.nanargmax(macro_f1s))
        self.optimal_threshold = thresholds[best_idx]

        c = self.COLORS.get(self.best_model_name, "#2563EB")
        ax.plot(thresholds, macro_f1s, color=c, lw=2.5)
        ax.axvline(self.optimal_threshold, color="#DC2626", ls="--", lw=1.5,
                   label=f"Optimal={self.optimal_threshold:.2f} "
                         f"(F1={macro_f1s[best_idx]:.3f})")
        ax.set_xlabel("Decision Threshold", fontsize=10)
        ax.set_ylabel("Macro F1", fontsize=10)
        ax.set_title("Precision/Recall Trade-off via Threshold",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    def _plot_feature_importance(self, ax):
        name  = ("XGBoost" if "XGBoost" in self.trainer.models
                 else "Random Forest")
        model = self.trainer.models[name]
        if not hasattr(model, "feature_importances_"):
            return
        fi      = model.feature_importances_
        n_emb   = self.extractor.embeddings.shape[1]
        labels  = ["[Embeddings]"] + self.extractor.numeric_cols
        vals    = np.concatenate([[fi[:n_emb].sum()], fi[n_emb:]])
        top_idx = vals.argsort()[::-1][:12]
        ax.barh([labels[i] for i in top_idx][::-1],
                vals[top_idx][::-1], color="#2563EB", alpha=0.8)
        ax.set_xlabel("Importance", fontsize=10)
        ax.set_title(f"Feature Importance – {name}",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

# 8.  ARTIFACT SAVER

class ArtifactSaver:
    """
    Definition

    Serialises all reusable pipeline objects to disk so the
    production scoring system can load them without re-training.

    Saved artifacts
    ─
    <model_name>_final.pkl   – best trained classifier
    scaler.pkl               – StandardScaler (for MLP inputs)
    pca.pkl                  – PCA projection
    hdbscan_clusterer.pkl    – fitted HDBSCAN object
    cluster_label_map.pkl    – {cluster_id: label} dict
    cluster_labels.parquet   – label assignments for the full dataset

    Step usage

    Called by HybridPipeline.run() as Step 8.

        saver = ArtifactSaver(config)
        saver.save(evaluator, trainer, extractor, clusterer, mapper, df_labeled)
    """

    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.cfg.output_dir.mkdir(exist_ok=True)

    def save(self,
             evaluator  : ModelEvaluator,
             trainer    : ModelTrainer,
             extractor  : FeatureExtractor,
             clusterer  : AlertClusterer,
             mapper     : LabelMapper,
             df_labeled : pd.DataFrame):
        best = evaluator.best_model_name
        joblib.dump(trainer.models[best],
                    self.cfg.output_dir / f"{best.replace(' ','_')}_final.pkl")
        joblib.dump(extractor.scaler,        self.cfg.output_dir / "scaler.pkl")
        joblib.dump(extractor.pca,           self.cfg.output_dir / "pca.pkl")
        joblib.dump(clusterer.clusterer,     self.cfg.output_dir / "hdbscan_clusterer.pkl")
        joblib.dump(mapper.label_map,        self.cfg.output_dir / "cluster_label_map.pkl")

        df_labeled[[self.cfg.label_col, "cluster_id", "cluster_strength"]].to_parquet(
            self.cfg.output_dir / "cluster_labels.parquet", index=True
        )
        log.info("Artifacts saved to %s/", self.cfg.output_dir)

# 9.  HYBRID PIPELINE  (orchestrator)
class HybridPipeline:
    """
    Definition

    Top-level orchestrator that wires all seven specialist classes
    into a single end-to-end workflow:

        FeatureExtractor  →  AlertClusterer  →  LabelMapper
        →  DataSplitter   →  ModelTrainer    →  HyperparameterTuner
        →  ModelEvaluator →  ArtifactSaver

    Instantiate with a DataFrame and an optional config, then call
    .run() to execute every step in sequence.

    Parameters

    df                 : location_disruption_df or equivalent
    config             : PipelineConfig (uses defaults if omitted)
    cluster_label_map  : pre-defined {cluster_id: label} dict.
                         If None, inspect() output must be reviewed
                         before labels are assigned.

    Step usage

        pipeline = HybridPipeline(df=location_disruption_df)
        pipeline.run()

    Or with customisation:
        cfg     = PipelineConfig(min_cluster_size=60, tuning_iter=20)
        labels  = {0: "signal_failure", 1: "weather", -1: "noise"}
        pipeline = HybridPipeline(df=location_disruption_df,
                                  config=cfg,
                                  cluster_label_map=labels)
        pipeline.run()
    """

    def __init__(self,
                 df: pd.DataFrame,
                 config: Optional[PipelineConfig] = None,
                 cluster_label_map: Optional[Dict[int, str]] = None):
        self.df                = df
        self.cfg               = config or PipelineConfig()
        self.cluster_label_map = cluster_label_map

        # Will be populated during run()
        self.extractor  : Optional[FeatureExtractor]     = None
        self.clusterer  : Optional[AlertClusterer]       = None
        self.mapper     : Optional[LabelMapper]          = None
        self.splitter   : Optional[DataSplitter]         = None
        self.trainer    : Optional[ModelTrainer]         = None
        self.tuner      : Optional[HyperparameterTuner]  = None
        self.evaluator  : Optional[ModelEvaluator]       = None
        self.saver      : Optional[ArtifactSaver]        = None

    #  public

    def run(self):
        """Execute all pipeline steps in sequence."""
        print("=" * 70)
        print("  Hybrid Clustering → Classification Pipeline")
        print("=" * 70)

        # Step 1 – Feature extraction
        print("\n[Step 1/8]  Feature Extraction")
        self.extractor = FeatureExtractor(self.cfg, self.df)
        X_full, X_pca  = self.extractor.extract()

        # Step 2 – Clustering
        print("\n[Step 2/8]  HDBSCAN Clustering")
        self.clusterer = AlertClusterer(self.cfg)
        df_clustered   = self.clusterer.fit_predict(self.extractor.df, X_pca)

        # Step 3 – Cluster inspection (always printed; manual labels can be
        #          passed via cluster_label_map at construction time)
        print("\n[Step 3/8]  Cluster Inspection")
        self.clusterer.inspect(df_clustered, self.extractor.text_col)

        # Step 4 – Label mapping
        print("\n[Step 4/8]  Label Mapping")
        self.mapper     = LabelMapper(self.cfg, self.cluster_label_map)
        df_labeled_all  = self.mapper.apply(df_clustered)
        df_labeled      = df_labeled_all[
            df_labeled_all[self.cfg.label_col] != "noise"
        ].copy()
        print(f"  Labeled rows: {len(df_labeled):,}  "
              f"(noise excluded: {(df_labeled_all[self.cfg.label_col]=='noise').sum():,})")

        # Step 5 – Train/val/test split + SMOTE
        print("\n[Step 5/8]  Data Splitting & SMOTE")
        self.splitter = DataSplitter(self.cfg)
        self.splitter.split(df_labeled, X_full)

        # Step 6 – Model training + cross-validation
        print("\n[Step 6/8]  Model Training & Cross-Validation")
        self.trainer = ModelTrainer(self.cfg, self.splitter)
        self.trainer.train_all()

        # Step 7 – Hyperparameter tuning
        print("\n[Step 7/8]  Hyperparameter Tuning")
        self.tuner = HyperparameterTuner(self.cfg, self.trainer, self.splitter)
        self.tuner.tune()

        # Step 8 – Evaluation
        print("\n[Step 8/8]  Final Evaluation")
        self.evaluator = ModelEvaluator(
            self.cfg, self.trainer, self.splitter, self.extractor
        )
        self.evaluator.evaluate()
        self.evaluator.plot()
        self.evaluator.print_summary()

        # Save artifacts
        self.saver = ArtifactSaver(self.cfg)
        self.saver.save(
            self.evaluator, self.trainer, self.extractor,
            self.clusterer, self.mapper, df_labeled,
        )

        gc.collect()
        print("\n✔  Pipeline complete.")
        return self


#  SAMPLE-FIT → FULL-INFER NLP PIPELINE
#
# Strategy for 176k+ rows
#
# Every model that requires *fitting* (BERTopic topic model, NER warm-up,
# language-detection calibration) is trained on a small stratified random
# sample drawn from the full corpus.  The fitted models are then applied
# to every row in the full dataset via efficient batched inference.
#
# Execution order
# ─
#   Step 0 — Language Detection   papluca/xlm-roberta-base-language-detection
#             • Run on sample  → learn language distribution
#             • Run on full    → tag every row with detected language + confidence
#             • Output: lang_code (category), lang_score (float32)
#
#   Step 1 — Named Entity Recognition + Geoparsing
#             • Model selected based on sample language profile:
#               - Majority non-Latin / highly multilingual → Babelscape/wikineural-multilingual-ner
#               - Latin-script multilingual               → Davlan/xlm-roberta-base-ner-hrl
#             • Run on full corpus in ner_chunk_size blocks
#             • Output: all_entities, loc_entities, first_loc_text, first_lat, first_lon
#
#   Step 2 — Sentiment Analysis   cardiffnlp/twitter-xlm-roberta-base-sentiment
#             • Batched DataLoader inference on full corpus
#             • OOM guard: batch size halved and retried on MemoryError
#             • Output: sentiment (category), score (float32)
#
#   Step 3 — Severity Assignment  (vectorised pandas, no GPU)
#             • Derived from sentiment × score thresholds
#             • Output: severity (category), severity_numeric (int8),
#                       alert_risk_level (category)
#
#   Step 4 — Topic Modeling       BERTopic + paraphrase-multilingual-MiniLM-L12-v2
#             • fit_transform() on stratified sample  → topic vocabulary
#             • transform() on full corpus in topic_chunk_size blocks
#             • Output: topic (int), topic_label (category), topic_words
#
# Sample strategy
# ─
# The sample is drawn once and reused across ALL fitting steps so that every
# model sees the same representative subset.  Stratification is on the first
# character of combined_text (a cheap proxy for script/language diversity).
# SAMPLE_SIZE default = 5 000: large enough to give BERTopic stable topics
# and the language detector a reliable distribution; small enough to be fast.
#
# Column contract (new columns added by this cell)
# ─
#   lang_code        str / category  — ISO 639-1 code, e.g. "nl", "en", "de"
#   lang_score       float32         — model confidence for detected language
#   all_entities     list[dict]      — all NER spans with type + score
#   loc_entities     list[str]       — location entity strings only
#   first_loc_text   str             — first resolved location name
#   first_lat        float32         — geocoded latitude  (NaN if not found)
#   first_lon        float32         — geocoded longitude (NaN if not found)
#   sentiment        category        — Positive / Neutral / Negative
#   score            float32         — sentiment confidence [0, 1]
#   severity         category        — low / medium / high / critical
#   severity_numeric int8            — 0 / 1 / 2 / 3
#   alert_risk_level category        — low / medium / high / critical
#   topic            int             — BERTopic topic id (-1 = outlier)
#   topic_label      category        — human-readable topic label
#   topic_words      str             — top keywords for the topic
#

import gc
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import pipeline as hf_pipeline

SEVERITY_MAP = {"low": 0, "medium": 1, "high": 2, "critical": 3}

#  NER model selection guide ─
# Ranked by multilingual coverage for transit alert corpora:
#   1. Babelscape/wikineural-multilingual-ner  — best coverage (16 languages,
#      Wikipedia-trained, handles mixed-script well); use when lang audit shows
#      Dutch + German + French + other non-English > 40% of corpus
#   2. Davlan/xlm-roberta-base-ner-hrl         — strong for high-resource Latin
#      languages (en/de/fr/nl/es/pt/ar); slightly faster than wikineural
#   3. xlm-roberta-large-finetuned-conll03     — highest accuracy on CoNLL-2003
#      benchmarks but only covers en/de/nl/es; use only if corpus is mostly
#      those four languages
# The pipeline auto-selects based on the sample language audit (see Step 0b).
NER_MODEL_MULTILINGUAL  = "Babelscape/wikineural-multilingual-ner"
NER_MODEL_HIGH_RESOURCE = "Davlan/xlm-roberta-base-ner-hrl"
NER_MODEL_CONLL         = "xlm-roberta-large-finetuned-conll03"

# Languages well-covered by the high-resource and CoNLL models
_HIGH_RESOURCE_LANGS = {"en", "de", "fr", "nl", "es", "pt", "ar", "it"}
_CONLL_LANGS         = {"en", "de", "nl", "es"}


#  Dataset ─
class AlertTextDataset(Dataset):
    """Minimal torch Dataset wrapping alert text strings for DataLoader batching."""
    def __init__(self, texts: list):
        self.texts = [str(t).strip() if t and str(t).strip() else "" for t in texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def _collate_texts(batch):
    """Identity collate — HuggingFace pipelines accept plain Python lists."""
    return batch


#  SampleFitFullInferPipeline
class SampleFitFullInferPipeline:
    """
    Four-step NLP pipeline optimised for large transit alert datasets.

    Core pattern

    1. Draw one stratified random sample (default 5 000 rows).
    2. Fit / calibrate every model that requires training on the sample.
    3. Run batched inference on the FULL dataset using the fitted models.

    This decouples fitting cost (O(sample_size)) from inference cost
    (O(n_full)), making the pipeline practical for 100k–1M row corpora.

    Parameters

    sample_size          : rows used for all fitting steps (default 5 000)
    lang_model           : HuggingFace model for language detection
    ner_model            : HuggingFace NER model; None = auto-select from
                           sample language audit
    sentiment_model      : HuggingFace sentiment model
    topic_embedding_model: sentence-transformers model for BERTopic
    device               : GPU index (0) or -1 for CPU; None = auto-detect
    ner_chunk_size       : rows per outer NER loop (memory safety margin)
    sentiment_batch_size : rows per sentiment DataLoader batch
    lang_batch_size      : rows per language detection DataLoader batch
    topic_chunk_size     : rows per BERTopic.transform() chunk
    num_workers          : DataLoader prefetch workers (0 on CPU/Windows)
    random_seed          : RNG seed for reproducible sampling
    """

    #  NER model auto-selection thresholds ─
    # If the top detected languages are all within the CoNLL-covered set AND
    # they account for > CONLL_COVERAGE_THRESHOLD of the sample → use CoNLL.
    # If top languages are all within high-resource set → use high-resource.
    # Otherwise → use the full multilingual model.
    CONLL_COVERAGE_THRESHOLD        = 0.90
    HIGH_RESOURCE_COVERAGE_THRESHOLD = 0.80

    def __init__(
        self,
        sample_size          : int           = 5_000,
        lang_model           : str           = "papluca/xlm-roberta-base-language-detection",
        ner_model            : str | None    = None,   # None = auto-select
        sentiment_model      : str           = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        topic_embedding_model: str           = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device               : int | None    = None,
        ner_chunk_size       : int           = 2_000,
        sentiment_batch_size : int           = 64,
        lang_batch_size      : int           = 128,
        topic_chunk_size     : int           = 10_000,
        num_workers          : int           = 2,
        random_seed          : int           = 42,
    ):
        import torch as _torch
        if device is None:
            device = 0 if _torch.cuda.is_available() else -1
        self.device = device

        gpu_name = _torch.cuda.get_device_name(device) if device >= 0 else "CPU"
        print(f"  Device: {gpu_name}")

        self.sample_size           = sample_size
        self.lang_model_name       = lang_model
        self._ner_model_override   = ner_model       # None = auto
        self.sentiment_model_name  = sentiment_model
        self.topic_embedding_model = topic_embedding_model
        self.ner_chunk_size        = ner_chunk_size
        self.sentiment_batch_size  = sentiment_batch_size
        self.lang_batch_size       = lang_batch_size
        self.topic_chunk_size      = topic_chunk_size
        self.num_workers           = num_workers if device >= 0 else 0
        self.rng                   = np.random.default_rng(random_seed)

        # Placeholders — loaded lazily after sample audit (NER model depends
        # on language distribution discovered in Step 0)
        self._lang_pipe    = None
        self._sent_pipe    = None
        self._ner_instance = None
        self._topic_model  = None
        self.ner_model_name = None   # set during _select_and_load_ner()
        self.lang_distribution: dict = {}   # populated in Step 0


    # PUBLIC INTERFACE

    def run(self, df: pd.DataFrame, text_column: str = "combined_text") -> pd.DataFrame:
        """
        Full pipeline: sample → fit → infer on all rows.

        Returns df with all NLP columns added (see module docstring for the
        complete column contract).
        """
        t_total = time.time()

        #  Sanitise
        stale = [
            "lang_code", "lang_score",
            "all_entities", "loc_entities", "first_loc_text",
            "first_lat", "first_lon",
            "sentiment", "score",
            "severity", "severity_numeric", "alert_risk_level",
            "topic", "topic_label", "topic_words",
        ]
        df = (df.drop(columns=[c for c in stale if c in df.columns], errors="ignore")
                .copy()
                .reset_index(drop=True))
        texts_arr = df[text_column].fillna("").astype(str).values

        #  Draw sample ─
        sample_idx = self._draw_sample(texts_arr)
        sample_texts = texts_arr[sample_idx].tolist()
        n_full   = len(texts_arr)
        n_sample = len(sample_idx)
        print(f"\n  Dataset : {n_full:,} rows")
        print(f"  Sample  : {n_sample:,} rows  (seed={self.rng.bit_generator.state['state']['state']})")

        #  Step 0: Language detection
        print("\n" + "─" * 60)
        print("  Step 0/4  Language Detection")
        print("─" * 60)
        df = self._run_language_detection(df, texts_arr, sample_texts)
        gc.collect()

        #  Step 1: NER + Geoparsing
        print("\n" + "─" * 60)
        print("  Step 1/4  Named Entity Recognition + Geoparsing")
        print("─" * 60)
        df = self._run_ner(df, text_column)
        gc.collect()

        #  Step 2: Sentiment Analysis
        print("\n" + "─" * 60)
        print("  Step 2/4  Sentiment Analysis")
        print("─" * 60)
        df = self._run_sentiment(df, texts_arr)
        gc.collect()

        #  Step 3: Severity Assignment ─
        print("\n" + "─" * 60)
        print("  Step 3/4  Severity Assignment  (vectorised)")
        print("─" * 60)
        df = self._assign_severity(df)

        #  Step 4: Topic Modeling
        print("\n" + "─" * 60)
        print("  Step 4/4  Topic Modeling")
        print("─" * 60)
        df = self._run_topics(df, texts_arr, sample_texts)
        gc.collect()

        elapsed = time.time() - t_total
        print(f"\n{'═'*60}")
        print(f"  ✓ Pipeline complete — {n_full:,} rows in {elapsed:.1f}s "
              f"({elapsed / n_full * 1_000:.1f} ms/row)")
        print(f"{'═'*60}")
        return df


    # SAMPLING

    def _draw_sample(self, texts_arr: np.ndarray) -> np.ndarray:
        """
        Stratified random sample of row indices.

        Stratification key: first character of each text, mapped to one of
        four script buckets (Latin / Cyrillic / CJK / other).  This ensures
        the sample reflects the script/language diversity of the full corpus
        without requiring a full language-detection pass first.

        Returns sorted array of integer indices into texts_arr.
        """
        n        = len(texts_arr)
        size     = min(self.sample_size, n)

        def _script_bucket(ch: str) -> str:
            o = ord(ch) if ch else 0
            if o < 0x0250:           return "latin"
            if 0x0400 <= o < 0x0500: return "cyrillic"
            if 0x4E00 <= o < 0xA000: return "cjk"
            return "other"

        buckets = np.array([
            _script_bucket(t[0]) if t else "other"
            for t in texts_arr
        ])
        unique_buckets, counts = np.unique(buckets, return_counts=True)
        proportions = counts / n

        idx_parts = []
        for bucket, prop in zip(unique_buckets, proportions):
            bucket_idx   = np.where(buckets == bucket)[0]
            n_pick       = max(1, round(size * prop))
            picked       = self.rng.choice(bucket_idx,
                                           size=min(n_pick, len(bucket_idx)),
                                           replace=False)
            idx_parts.append(picked)

        sample_idx = np.sort(np.concatenate(idx_parts))
        # Trim/top-up to exact size
        if len(sample_idx) > size:
            sample_idx = sample_idx[:size]
        elif len(sample_idx) < size:
            remaining = np.setdiff1d(np.arange(n), sample_idx)
            extra     = self.rng.choice(remaining,
                                        size=size - len(sample_idx),
                                        replace=False)
            sample_idx = np.sort(np.concatenate([sample_idx, extra]))

        bucket_summary = {b: int(c) for b, c in zip(unique_buckets, counts)}
        print(f"    Script buckets in full corpus : {bucket_summary}")
        print(f"    Sample size drawn             : {len(sample_idx):,}")
        return sample_idx


    # STEP 0 — LANGUAGE DETECTION

    def _run_language_detection(
        self,
        df         : pd.DataFrame,
        texts_arr  : np.ndarray,
        sample_texts: list,
    ) -> pd.DataFrame:
        """
        Model  : papluca/xlm-roberta-base-language-detection
        Fit on : sample  → build language distribution, select NER model
        Infer  : full corpus in lang_batch_size chunks

        Why detect language first?
          The NER model is selected based on which languages dominate the
          corpus.  Running language detection on the full 176k rows is fast
          because the model is lightweight (XLM-RoBERTa base, classification
          head only — no generation, no sequence labelling overhead).
        """
        print(f"    Model  : {self.lang_model_name}")
        print(f"    Fitting on {len(sample_texts):,} sample rows …")

        # Load language detection pipeline
        self._lang_pipe = hf_pipeline(
            "text-classification",
            model     = self.lang_model_name,
            device    = self.device,
            truncation= True,
            max_length= 128,
            top_k     = 1,
        )

        #  0a: Run on sample to build language distribution
        sample_results = self._batched_lang_infer(sample_texts, self.lang_batch_size)
        sample_langs   = [r[0]["label"] for r in sample_results]

        lang_counts = pd.Series(sample_langs).value_counts(normalize=True)
        self.lang_distribution = lang_counts.head(10).to_dict()
        print(f"\n    Sample language distribution (top 10):")
        for lang, pct in self.lang_distribution.items():
            bar = "█" * int(pct * 40)
            print(f"      {lang:6}  {pct:5.1%}  {bar}")

        #  0b: Select NER model based on language profile
        self._select_and_load_ner(lang_counts)

        #  0c: Run language detection on FULL corpus
        print(f"\n    Inferring on full {len(texts_arr):,} rows …")
        all_results = self._batched_lang_infer(texts_arr.tolist(), self.lang_batch_size,
                                               show_progress=True, total=len(texts_arr))

        df["lang_code"]  = pd.Categorical([r[0]["label"] for r in all_results])
        df["lang_score"] = np.array([r[0]["score"] for r in all_results],
                                    dtype=np.float32)

        print(f"\n    Full corpus language distribution:")
        print(df["lang_code"].value_counts().head(10).to_string())
        print(f"    Avg detection confidence: {df['lang_score'].mean():.3f}")
        return df

    def _batched_lang_infer(
        self,
        texts        : list,
        batch_size   : int,
        show_progress: bool = False,
        total        : int  = 0,
    ) -> list:
        """Run language detection pipeline in batch_size chunks; return raw results."""
        results = []
        done    = 0
        for start in range(0, len(texts), batch_size):
            batch   = texts[start: start + batch_size]
            out     = self._lang_pipe(batch)
            # out is list of list-of-dicts when top_k > 0
            results.extend(out if isinstance(out[0], list) else [[r] for r in out])
            done += len(batch)
            if show_progress:
                print(f"    Lang {done:>7,}/{total:,} …", end="\r")
        return results

    def _select_and_load_ner(self, lang_counts: pd.Series) -> None:
        """
        Choose the NER model based on sample language coverage, then load it.

        Decision logic:
          ① If override was given at construction → use that model exactly.
          ② If ≥ CONLL_COVERAGE_THRESHOLD of texts are in {en,de,nl,es}
             → xlm-roberta-large-finetuned-conll03  (highest accuracy for those 4)
          ③ If ≥ HIGH_RESOURCE_COVERAGE_THRESHOLD in the 8-language high-resource set
             → Davlan/xlm-roberta-base-ner-hrl       (fast, good coverage)
          ④ Otherwise
             → Babelscape/wikineural-multilingual-ner (broadest coverage)
        """
        if self._ner_model_override is not None:
            chosen = self._ner_model_override
            reason = "manual override"
        else:
            top_langs    = set(lang_counts.index.tolist())
            conll_cov    = lang_counts[lang_counts.index.isin(_CONLL_LANGS)].sum()
            hr_cov       = lang_counts[lang_counts.index.isin(_HIGH_RESOURCE_LANGS)].sum()

            if conll_cov >= self.CONLL_COVERAGE_THRESHOLD:
                chosen = NER_MODEL_CONLL
                reason = f"CoNLL-4 coverage = {conll_cov:.0%} ≥ {self.CONLL_COVERAGE_THRESHOLD:.0%}"
            elif hr_cov >= self.HIGH_RESOURCE_COVERAGE_THRESHOLD:
                chosen = NER_MODEL_HIGH_RESOURCE
                reason = f"high-resource coverage = {hr_cov:.0%} ≥ {self.HIGH_RESOURCE_COVERAGE_THRESHOLD:.0%}"
            else:
                chosen = NER_MODEL_MULTILINGUAL
                reason = f"multilingual corpus (high-resource coverage only {hr_cov:.0%})"

        self.ner_model_name = chosen
        print(f"\n    NER model selected : {chosen}")
        print(f"    Reason             : {reason}")

        self._ner_instance = MultilingualNER(
            model_name       = chosen,
            device           = self.device,
            batch_size       = self.ner_chunk_size,
            enable_geocoding = True,
        )

    # STEP 1 — NER + GEOPARSING

    def _run_ner(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        NER + geoparsing on the FULL corpus in ner_chunk_size blocks.

        Model already selected and loaded in Step 0b.
        Outer loop provides memory safety and progress reporting;
        the inner geoparse_to_dataframe() handles GPU batching internally.
        """
        print(f"    Model  : {self.ner_model_name}")
        print(f"    Chunks : {self.ner_chunk_size:,} rows each")

        total     = len(df)
        geo_parts = []

        for start in range(0, total, self.ner_chunk_size):
            end   = min(start + self.ner_chunk_size, total)
            chunk = df.iloc[start:end].copy()
            chunk = self._ner_instance.geoparse_to_dataframe(
                chunk, text_column=text_column, min_score=0.5
            )
            geo_parts.append(chunk)
            print(f"    NER {end:>7,}/{total:,} rows …", end="\r")

        df = pd.concat(geo_parts, ignore_index=True)
        n_geo = df["first_lat"].notna().sum()
        print(f"\n    Complete — {n_geo:,} / {total:,} rows geocoded "
              f"({n_geo/total:.1%})")
        return df

    # STEP 2 — SENTIMENT ANALYSIS

    def _run_sentiment(self, df: pd.DataFrame, texts_arr: np.ndarray) -> pd.DataFrame:
        """
        Sentiment on the FULL corpus via DataLoader.

        Model: cardiffnlp/twitter-xlm-roberta-base-sentiment
        Supports 20+ languages; trained on Twitter multilingual data — good
        fit for short, informal transit alert text.

        DataLoader batching
          Each batch is fed to sentiment.analyze_batch() in one GPU forward
          pass.  The DataLoader prefetches the next batch on CPU workers while
          the GPU processes the current one (pin_memory DMA transfer).

        OOM guard
          If a batch raises MemoryError the batch size is halved and the batch
          is retried once.  This handles GPU fragmentation on long-running
          notebooks without crashing the pipeline.
        """
        print(f"    Model  : {self.sentiment_model_name}")

        # Load sentiment model on first call
        if self._sent_pipe is None:
            self._sent_pipe = MultilingualSentiment(
                model_name = self.sentiment_model_name,
                device     = self.device,
                batch_size = self.sentiment_batch_size,
            )

        dataset    = AlertTextDataset(texts_arr.tolist())
        loader     = DataLoader(
            dataset,
            batch_size  = self.sentiment_batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            collate_fn  = _collate_texts,
            pin_memory  = (self.device >= 0),
        )

        all_results = []
        done        = 0
        total       = len(texts_arr)
        batch_size  = self.sentiment_batch_size

        for batch_texts in loader:
            try:
                result = self._sent_pipe.analyze_batch(
                    batch_texts, batch_size=batch_size, show_progress=False
                )
            except MemoryError:
                batch_size = max(1, batch_size // 2)
                print(f"\n    OOM — retrying with batch_size={batch_size}")
                result = self._sent_pipe.analyze_batch(
                    batch_texts, batch_size=batch_size, show_progress=False
                )
            all_results.append(result)
            done += len(batch_texts)
            print(f"    Sentiment {done:>7,}/{total:,} …", end="\r")

        sent_df = pd.concat(all_results, ignore_index=True)
        df["sentiment"] = sent_df["sentiment"].astype("category").values
        df["score"]     = pd.to_numeric(sent_df["score"],
                                        errors="coerce").astype("float32").values

        print(f"\n    Distribution:\n{df['sentiment'].value_counts().to_string()}")
        return df

    # STEP 3 — SEVERITY ASSIGNMENT  (no model — vectorised pandas)

    def _assign_severity(self, df: pd.DataFrame) -> pd.DataFrame:
        sent_lower = df["sentiment"].astype(str).str.lower()
        df["severity"] = "low"
        neg = sent_lower == "negative"
        df.loc[neg & (df["score"] >= 0.85), "severity"] = "critical"
        df.loc[neg & (df["score"] >= 0.65) & (df["score"] < 0.85), "severity"] = "high"
        df.loc[neg & (df["score"] >= 0.40) & (df["score"] < 0.65), "severity"] = "medium"
        df["severity"]         = df["severity"].astype("category")
        df["severity_numeric"] = df["severity"].map(SEVERITY_MAP).astype("int8")
        df["alert_risk_level"] = pd.cut(
            df["severity_numeric"],
            bins=[0, 1, 2, 3, 4],
            labels=["low", "medium", "high", "critical"],
            include_lowest=True,
        ).astype("category")
        print(f"    Severity:\n{df['severity'].value_counts().to_string()}")
        print(f"\n    Sentiment × Severity:")
        print(pd.crosstab(df["sentiment"], df["severity"]))
        return df

    # STEP 4 — TOPIC MODELING

    def _run_topics(
        self,
        df          : pd.DataFrame,
        texts_arr   : np.ndarray,
        sample_texts: list,
    ) -> pd.DataFrame:
        """
        BERTopic with paraphrase-multilingual-MiniLM-L12-v2 embeddings.

        Fit pattern (sample → full)
        ─
        fit_transform(sample_texts)
          → embeds sample_size texts through MiniLM-L12-v2
          → runs UMAP dimensionality reduction on sample embeddings
          → clusters with HDBSCAN
          → builds c-TF-IDF topic representations
          Total embedding cost: O(sample_size)  ~5 000 texts ≈ 15–30 s on GPU

        transform(full_texts)  [chunked]
          → embeds each chunk through MiniLM-L12-v2
          → assigns each text to nearest topic centroid (no UMAP/HDBSCAN refit)
          Total embedding cost: O(n_full)  176k texts ≈ 3–8 min on GPU

        Why sample-fit?
          Full fit on 176k texts: UMAP on 176k × 384 matrix → OOM on <32 GB RAM.
          Sample fit on 5k texts: UMAP on 5k × 384 matrix → ~200 MB, fast.
          Topic quality is stable for sample_size ≥ 2 000 on transit corpora.
        """
        print(f"    Embedding model : {self.topic_embedding_model}")
        print(f"    Fit on sample   : {len(sample_texts):,} texts")
        print(f"    Infer on full   : {len(texts_arr):,} texts  "
              f"(chunks of {self.topic_chunk_size:,})")

        if self._topic_model is None:
            self._topic_model = MultilingualTopicModeling(
                language              = "multilingual",
                min_topic_size        = 15,
                embedding_model       = self.topic_embedding_model,
            )

        #  A: Fit on sample
        self._topic_model.fit_transform(
            sample_texts,
            sample_size = len(sample_texts),
        )
        n_topics = len(self._topic_model.label_map)
        print(f"    Topics discovered in sample : {n_topics}")

        #  B: Transform full corpus in chunks
        texts_list = texts_arr.tolist()
        all_topics: list = []
        total = len(texts_list)

        for start in range(0, total, self.topic_chunk_size):
            end   = min(start + self.topic_chunk_size, total)
            t, _  = self._topic_model.topic_model.transform(texts_list[start:end])
            all_topics.extend(t)
            print(f"    Topics {end:>7,}/{total:,} …", end="\r")

        print(f"\n    Inference complete.")

        #  C: Attach columns ─
        df["topic"]       = all_topics
        df["topic_label"] = (df["topic"].map(self._topic_model.label_map)
                               .fillna("Outlier").astype("category"))
        df["topic_words"] = df["topic"].map(self._topic_model.words_map)

        #  D: Diagnostics
        self._topic_model.analyze_topics(all_topics, texts_list)
        outlier_pct = (df["topic"] == -1).mean()
        print(f"\n    Outlier rows (topic = -1) : {outlier_pct:.1%}")
        print(f"    Top 10 topic labels:\n"
              f"{df['topic_label'].value_counts().head(10).to_string()}")
        return df

class GPUAcceleratedNLPPipeline:
    """
    Optimized GPU NLP pipeline for large datasets.
    Performs:
    - NER + geoparsing
    - sentiment analysis
    - severity scoring
    - topic modeling (sample fit + full inference)
    """

    def __init__(
        self,
        ner_model: str = "Davlan/xlm-roberta-base-ner-hrl",
        sentiment_model: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        topic_language: str = "multilingual",
        device: int = 0,
        ner_batch_size: int = 256,
        sentiment_batch_size: int = 256,
        topic_sample_size: int = 20000,
    ):

        # NER + geoparsing
        self.ner = MultilingualNER(
            model_name=ner_model,
            device=device,
            batch_size=ner_batch_size,
            enable_geocoding=True,
        )

        # Sentiment
        self.sentiment = MultilingualSentiment(
            model_name=sentiment_model,
            device=device
        )

        self.sentiment_batch_size = sentiment_batch_size

        # Topic modeling
        self.topic_model = MultilingualTopicModeling(language=topic_language)
        self.topic_sample_size = topic_sample_size

        # batch configs
        self.ner_batch_size = ner_batch_size

    def run(self, df: pd.DataFrame, text_column: str = "combined_text") -> pd.DataFrame:

        print("Preparing text")

        texts = (
            df[text_column]
            .fillna("")
            .astype(str)
            .values
        )

        # Step 1: NER + Geoparsing (GPU batched internally)

        print("Step 1: NER + Geoparsing")

        df = self.ner.geoparse_to_dataframe(
            df,
            text_column=text_column,
            batch_size=self.ner_batch_size
        )

        # Step 2: Sentiment Analysis (single GPU call)

        print("Step 2: Sentiment Analysis")

        sent_df = self.sentiment.analyze_batch(
            texts.tolist(),
            batch_size=self.sentiment_batch_size,
            show_progress=True
        )

        df["sentiment"] = sent_df["sentiment"].values
        df["score"] = sent_df["score"].astype("float32").values

        # Step 3: Vectorized Severity Scoring

        print("Step 3: Severity scoring")

        score = df["score"].values
        sentiment = df["sentiment"].values

        severity = np.full(len(df), "low", dtype=object)

        neg = sentiment == "negative"

        severity[neg & (score >= 0.85)] = "critical"
        severity[neg & (score >= 0.65) & (score < 0.85)] = "high"
        severity[neg & (score >= 0.40) & (score < 0.65)] = "medium"

        df["severity"] = pd.Categorical(
            severity,
            categories=["low", "medium", "high", "critical"]
        )

        df["severity_numeric"] = df["severity"].map(SEVERITY_MAP).astype("int8")

        df["alert_risk_level"] = pd.cut(
            df["severity_numeric"],
            bins=[0, 1, 2, 3, 4],
            labels=["low", "medium", "high", "critical"],
            include_lowest=True,
        ).astype("category")

        # Step 4: Topic Modeling (sample fit → full inference)

        print("Step 4: Topic modeling")

        sample_size = min(self.topic_sample_size, len(texts))

        sample_idx = np.random.default_rng(42).choice(
            len(texts),
            size=sample_size,
            replace=False
        )

        sample_texts = texts[sample_idx].tolist()

        self.topic_model.fit_transform(sample_texts)

        topics_full, _ = self.topic_model.topic_model.transform(texts.tolist())

        df["topic"] = topics_full

        df["topic_label"] = (
            pd.Series(topics_full)
            .map(self.topic_model.label_map)
            .astype("category")
        )

        df["topic_words"] = pd.Series(topics_full).map(
            self.topic_model.words_map
        )

        print("Pipeline completed")

        return df

def stratified_sample_lang_composite_severity(
    df,
    n=2000,
    text_col="combined_text",
    lang_col="language",
    binary_targets=None,      # list of binary target column names
    severity_col="severity_numeric",
    random_state=42
):
    """
    Stratification key: language × composite_disruption × severity_tier
    Produces 4 × 2 × 2 = 16 strata maximum.

    - language        : Dutch / German / English / French+other
    - composite flag  : 1 if ANY binary target is positive, else 0
    - severity tier   : 0 = low (severity_numeric == 0)
                        1 = medium-high (severity_numeric >= 1)
    """
    if binary_targets is None:
        binary_targets = [
            "early_warning_target",
            "disruption_target",
            "future_alert",
        ]

    df = df.copy()

    #  1. Language bucket
    if lang_col in df.columns and df[lang_col].notna().mean() > 0.5:
        lang_map = {
            "nl": "dutch",
            "de": "german",
            "en": "english",
            "fr": "french",
        }
        df["_lang"] = (
            df[lang_col]
            .map(lang_map)
            .fillna("other")          # all other languages → "other"
        )
        print(f"✓ Language key from column '{lang_col}'")
    else:
        # Pre-NLP fallback: character script proxy
        def script_bucket(text):
            if not isinstance(text, str) or len(text) == 0:
                return "other"
            cp = ord(text[0])
            if 0x0041 <= cp <= 0x024F: return "latin"   # covers nl/de/en/fr
            if 0x0400 <= cp <= 0x04FF: return "cyrillic"
            if 0x4E00 <= cp <= 0x9FFF: return "cjk"
            if 0x0600 <= cp <= 0x06FF: return "arabic"
            return "other"
        df["_lang"] = df[text_col].apply(script_bucket)
        print("✓ Language key from character script proxy (pre-NLP fallback)")

    #  2. Composite disruption flag
    active_binary = [
        col for col in binary_targets
        if col in df.columns and df[col].notna().mean() > 0.5
    ]

    if active_binary:
        df["_composite"] = (
            df[active_binary]
            .fillna(0)
            .astype(int)
            .max(axis=1)              # 1 if ANY target is positive
            .astype(str)
        )
        print(f"✓ Composite flag from: {active_binary}")
    else:
        df["_composite"] = "0"
        print("⚠ No binary targets found — composite flag set to 0 for all rows")

    #  3. Severity tier
    if severity_col in df.columns and df[severity_col].notna().mean() > 0.5:
        df["_severity_tier"] = (
            df[severity_col]
            .fillna(0)
            .astype(int)
            .clip(lower=0)
            .apply(lambda x: "high" if x >= 1 else "low")
        )
        print(f"✓ Severity tier from column '{severity_col}'")
    else:
        df["_severity_tier"] = "low"
        print(f"⚠ '{severity_col}' not found — severity tier set to 'low' for all rows")

    #  4. Combined stratification key ─
    df["_strat_key"] = (
        df["_lang"]
        + "__" + df["_composite"]
        + "__" + df["_severity_tier"]
    )

    strata_counts = df["_strat_key"].value_counts().sort_index()

    print(f"\n{'─'*55}")
    print(f"  Strata breakdown (max possible: 4×2×2 = 16)")
    print(f"{'─'*55}")
    print(f"  {'Stratum':<35} {'Count':>8}  {'% of corpus':>12}")
    print(f"{'─'*55}")
    for stratum, count in strata_counts.items():
        pct = count / len(df) * 100
        print(f"  {stratum:<35} {count:>8,}  {pct:>11.1f}%")
    print(f"{'─'*55}")
    print(f"  {'TOTAL':<35} {len(df):>8,}  {'100.0%':>12}")
    print(f"{'─'*55}")

    #  5. Proportional allocation
    allocation = (strata_counts / len(df) * n).apply(lambda x: max(1, round(x)))

    # Trim/top-up to hit exactly n
    diff = int(n - allocation.sum())
    indices = list(strata_counts.index)
    i = 0
    while diff > 0:
        allocation[indices[i % len(indices)]] += 1
        diff -= 1
        i += 1
    while diff < 0:
        key = indices[i % len(indices)]
        if allocation[key] > 1:
            allocation[key] -= 1
            diff += 1
        i += 1

    #  6. Draw sample from each stratum
    parts = []
    for stratum, n_draw in allocation.items():
        group = df[df["_strat_key"] == stratum]
        actual_draw = min(n_draw, len(group))
        if actual_draw < n_draw:
            print(f"  ⚠ Stratum '{stratum}': requested {n_draw}, only {actual_draw} available")
        parts.append(group.sample(n=actual_draw, random_state=random_state))

    sample = (
        pd.concat(parts)
        .sample(frac=1, random_state=random_state)   # shuffle
        .reset_index(drop=True)
        .drop(columns=["_lang", "_composite", "_severity_tier", "_strat_key"])
    )

    #  7. Verify preservation
    print(f"\n✓ Sample shape: {sample.shape}")
    print(f"\n{'─'*55}")
    print("  Distribution check — full corpus vs sample")
    print(f"{'─'*55}")

    check_cols = active_binary + (
        [severity_col] if severity_col in sample.columns else []
    )
    for col in check_cols:
        if col not in sample.columns:
            continue
        full_pct   = df[col].value_counts(normalize=True).mul(100).round(1)
        sample_pct = sample[col].value_counts(normalize=True).mul(100).round(1)
        comparison = pd.DataFrame({
            "Full %":   full_pct,
            "Sample %": sample_pct
        }).fillna(0.0)
        print(f"\n  {col}:\n{comparison.to_string()}")

    return sample


class GPUAcceleratedNLPPipeline:
    """
    Optimized GPU NLP pipeline for large datasets.
    Performs:
    - NER + geoparsing
    - sentiment analysis
    - severity scoring
    - topic modeling (sample fit + full inference)
    """

    def __init__(
        self,
        ner_model: str = "Davlan/xlm-roberta-base-ner-hrl",
        sentiment_model: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        topic_language: str = "multilingual",
        device: int = 0,
        ner_batch_size: int = 256,
        sentiment_batch_size: int = 256,
        topic_sample_size: int = 20000,
    ):

        # NER + geoparsing
        self.ner = MultilingualNER(
            model_name=ner_model,
            device=device,
            batch_size=ner_batch_size,
            enable_geocoding=True,
        )

        # Sentiment
        self.sentiment = MultilingualSentiment(
            model_name=sentiment_model,
            device=device
        )

        self.sentiment_batch_size = sentiment_batch_size

        # Topic modeling
        self.topic_model = MultilingualTopicModeling(language=topic_language)
        self.topic_sample_size = topic_sample_size

        # batch configs
        self.ner_batch_size = ner_batch_size

    def run(self, df: pd.DataFrame, text_column: str = "combined_text") -> pd.DataFrame:

        print("Preparing text")

        texts = (
            df[text_column]
            .fillna("")
            .astype(str)
            .values
        )


        # Step 1: NER + Geoparsing (GPU batched internally)

        print("Step 1: NER + Geoparsing")

        df = self.ner.geoparse_to_dataframe(
            df,
            text_column=text_column,
            batch_size=self.ner_batch_size
        )


        # Step 2: Sentiment Analysis (single GPU call)

        print("Step 2: Sentiment Analysis")

        sent_df = self.sentiment.analyze_batch(
            texts.tolist(),
            batch_size=self.sentiment_batch_size,
            show_progress=True
        )

        df["sentiment"] = sent_df["sentiment"].values
        df["score"] = sent_df["score"].astype("float32").values


        # Step 3: Vectorized Severity Scoring

        print("Step 3: Severity scoring")

        score = df["score"].values
        sentiment = df["sentiment"].values

        severity = np.full(len(df), "low", dtype=object)

        neg = sentiment == "negative"

        severity[neg & (score >= 0.85)] = "critical"
        severity[neg & (score >= 0.65) & (score < 0.85)] = "high"
        severity[neg & (score >= 0.40) & (score < 0.65)] = "medium"

        df["severity"] = pd.Categorical(
            severity,
            categories=["low", "medium", "high", "critical"]
        )

        df["severity_numeric"] = df["severity"].map(SEVERITY_MAP).astype("int8")

        df["alert_risk_level"] = pd.cut(
            df["severity_numeric"],
            bins=[0, 1, 2, 3, 4],
            labels=["low", "medium", "high", "critical"],
            include_lowest=True,
        ).astype("category")


        # Step 4: Topic Modeling (sample fit → full inference)

        print("Step 4: Topic modeling")

        sample_size = min(self.topic_sample_size, len(texts))

        sample_idx = np.random.default_rng(42).choice(
            len(texts),
            size=sample_size,
            replace=False
        )

        sample_texts = texts[sample_idx].tolist()

        self.topic_model.fit_transform(sample_texts)

        topics_full, _ = self.topic_model.topic_model.transform(texts.tolist())

        df["topic"] = topics_full

        df["topic_label"] = (
            pd.Series(topics_full)
            .map(self.topic_model.label_map)
            .astype("category")
        )

        df["topic_words"] = pd.Series(topics_full).map(
            self.topic_model.words_map
        )

        print("Pipeline completed")

        return df

