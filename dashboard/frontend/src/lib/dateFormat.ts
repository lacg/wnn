/**
 * Date format detection and formatting utilities.
 *
 * Detects the user's preferred date/time format by examining what the browser
 * actually produces via toLocaleString(), then uses that detected format for
 * consistent formatting throughout the app.
 *
 * Supports:
 * - Date orders: yyyy-MM-dd (ISO), dd/MM/yyyy, MM/dd/yyyy
 * - Separators: / . - (slash, dot, dash)
 * - Padding: 01/05/2026 vs 1/5/2026
 * - Time: 24h vs 12h
 */

export interface DateFormatPrefs {
  order: 'ymd' | 'dmy' | 'mdy';
  separator: string;
  padded: boolean;
  is24h: boolean;
}

const STORAGE_KEY = 'dateFormatPrefs';
let cachedPrefs: DateFormatPrefs | null = null;

/**
 * Get user's date format preferences.
 * Priority: localStorage override > auto-detected from browser
 */
export function detectDateFormat(): DateFormatPrefs {
  if (cachedPrefs) return cachedPrefs;

  // Check for user override in localStorage
  if (typeof localStorage !== 'undefined') {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        const parsed = JSON.parse(stored) as DateFormatPrefs;
        if (parsed.order && parsed.separator) {
          cachedPrefs = parsed;
          return cachedPrefs;
        }
      } catch {
        // Invalid stored prefs, fall through to auto-detect
      }
    }
  }

  // Use 2019-07-25 14:30 - all components are distinguishable:
  // - 2019/19 = year (only value that could be 4-digit or match 19)
  // - 25 = day (only value > 12, can't be month)
  // - 7/07 = month (remaining value)
  const testDate = new Date(2019, 6, 25, 14, 30);
  const dateStr = testDate.toLocaleDateString();
  const timeStr = testDate.toLocaleTimeString();

  // Extract separator (first delimiter between numbers)
  const sepMatch = dateStr.match(/\d([\/\.\-\s])\d/);
  const separator = sepMatch ? sepMatch[1] : '/';

  // Extract all number groups
  const numbers = dateStr.match(/\d+/g)?.map(n => parseInt(n, 10)) || [];

  // Determine component order
  let order: 'ymd' | 'dmy' | 'mdy' = 'dmy';
  if (numbers.length >= 3) {
    // Find year index (2019 or 19)
    let yearIdx = numbers.findIndex(n => n === 2019);
    if (yearIdx === -1) yearIdx = numbers.findIndex(n => n === 19);

    // Find day index (25 - the only value > 12)
    const dayIdx = numbers.findIndex(n => n === 25);

    // Month is the remaining index
    const monthIdx = [0, 1, 2].find(i => i !== yearIdx && i !== dayIdx) ?? 1;

    // Build order string based on positions
    if (yearIdx === 0) {
      order = 'ymd'; // ISO format
    } else if (dayIdx === 0) {
      order = 'dmy'; // European
    } else if (monthIdx === 0) {
      order = 'mdy'; // US
    }
  }

  // Detect zero-padding by checking if month appears as "07"
  const padded = dateStr.includes('07');

  // Detect 12h vs 24h by checking for AM/PM markers
  const is24h = !timeStr.toLowerCase().includes('am') &&
                !timeStr.toLowerCase().includes('pm');

  cachedPrefs = { order, separator, padded, is24h };
  return cachedPrefs;
}

/**
 * Format a date string according to detected preferences.
 */
export function formatDate(dateStr: string | null): string {
  if (!dateStr) return '-';

  const date = new Date(dateStr);
  if (isNaN(date.getTime())) return '-';

  const prefs = detectDateFormat();
  return formatDateWithPrefs(date, prefs);
}

/**
 * Format a Date object according to given preferences.
 */
export function formatDateWithPrefs(date: Date, prefs: DateFormatPrefs): string {
  const { order, separator, padded, is24h } = prefs;

  const d = date.getDate();
  const m = date.getMonth() + 1;
  const y = date.getFullYear();

  const day = padded ? d.toString().padStart(2, '0') : d.toString();
  const month = padded ? m.toString().padStart(2, '0') : m.toString();
  const year = y.toString();

  // Build date part based on order
  let datePart: string;
  switch (order) {
    case 'ymd':
      datePart = `${year}${separator}${month}${separator}${day}`;
      break;
    case 'mdy':
      datePart = `${month}${separator}${day}${separator}${year}`;
      break;
    case 'dmy':
    default:
      datePart = `${day}${separator}${month}${separator}${year}`;
      break;
  }

  // Build time part
  let hours = date.getHours();
  const minutes = date.getMinutes().toString().padStart(2, '0');
  let timePart: string;

  if (is24h) {
    timePart = `${hours.toString().padStart(2, '0')}:${minutes}`;
  } else {
    const ampm = hours >= 12 ? 'PM' : 'AM';
    hours = hours % 12 || 12;
    timePart = `${hours}:${minutes} ${ampm}`;
  }

  return `${datePart}, ${timePart}`;
}

/**
 * Format just the date part (no time).
 */
export function formatDateOnly(dateStr: string | null): string {
  if (!dateStr) return '-';

  const date = new Date(dateStr);
  if (isNaN(date.getTime())) return '-';

  const prefs = detectDateFormat();
  const { order, separator, padded } = prefs;

  const d = date.getDate();
  const m = date.getMonth() + 1;
  const y = date.getFullYear();

  const day = padded ? d.toString().padStart(2, '0') : d.toString();
  const month = padded ? m.toString().padStart(2, '0') : m.toString();
  const year = y.toString();

  switch (order) {
    case 'ymd':
      return `${year}${separator}${month}${separator}${day}`;
    case 'mdy':
      return `${month}${separator}${day}${separator}${year}`;
    case 'dmy':
    default:
      return `${day}${separator}${month}${separator}${year}`;
  }
}

/**
 * Format just the time part.
 */
export function formatTimeOnly(dateStr: string | null): string {
  if (!dateStr) return '-';

  const date = new Date(dateStr);
  if (isNaN(date.getTime())) return '-';

  const { is24h } = detectDateFormat();

  let hours = date.getHours();
  const minutes = date.getMinutes().toString().padStart(2, '0');

  if (is24h) {
    return `${hours.toString().padStart(2, '0')}:${minutes}`;
  } else {
    const ampm = hours >= 12 ? 'PM' : 'AM';
    hours = hours % 12 || 12;
    return `${hours}:${minutes} ${ampm}`;
  }
}

/**
 * Get a human-readable description of the detected format.
 * Useful for debugging or showing in settings.
 */
export function getFormatDescription(): string {
  const prefs = detectDateFormat();

  const orderNames = {
    ymd: 'YYYY-MM-DD (ISO)',
    dmy: 'DD/MM/YYYY (European)',
    mdy: 'MM/DD/YYYY (US)'
  };

  const sepNames: Record<string, string> = {
    '/': 'slash',
    '.': 'dot',
    '-': 'dash',
    ' ': 'space'
  };

  return `${orderNames[prefs.order]} with ${sepNames[prefs.separator] || prefs.separator}, ` +
         `${prefs.padded ? 'zero-padded' : 'no padding'}, ` +
         `${prefs.is24h ? '24-hour' : '12-hour'} time`;
}

/**
 * Save user's preferred date format to localStorage.
 * Clears cache so new format takes effect immediately.
 */
export function savePreferences(prefs: DateFormatPrefs): void {
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
  }
  cachedPrefs = prefs;
}

/**
 * Clear saved preferences and revert to auto-detection.
 */
export function clearPreferences(): void {
  if (typeof localStorage !== 'undefined') {
    localStorage.removeItem(STORAGE_KEY);
  }
  cachedPrefs = null;
}

/**
 * Check if user has custom preferences saved.
 */
export function hasCustomPreferences(): boolean {
  if (typeof localStorage === 'undefined') return false;
  return localStorage.getItem(STORAGE_KEY) !== null;
}

/**
 * Get auto-detected preferences (ignoring any saved override).
 * Useful for showing "detected" vs "custom" in settings.
 */
export function getAutoDetectedPrefs(): DateFormatPrefs {
  // Temporarily clear cache to force re-detection
  const saved = cachedPrefs;
  cachedPrefs = null;

  // Remove stored prefs temporarily
  let storedValue: string | null = null;
  if (typeof localStorage !== 'undefined') {
    storedValue = localStorage.getItem(STORAGE_KEY);
    localStorage.removeItem(STORAGE_KEY);
  }

  // Detect
  const detected = detectDateFormat();

  // Restore
  if (typeof localStorage !== 'undefined' && storedValue) {
    localStorage.setItem(STORAGE_KEY, storedValue);
  }
  cachedPrefs = saved;

  return detected;
}

/**
 * Format a sample date for preview purposes.
 */
export function formatSampleDate(prefs: DateFormatPrefs): string {
  const sampleDate = new Date(2026, 0, 28, 14, 30); // Jan 28, 2026, 14:30
  return formatDateWithPrefs(sampleDate, prefs);
}
