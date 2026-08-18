/**
 * Small fetch helpers shared by pages.
 */

/**
 * GET a JSON endpoint. Throws on !ok, using the server's JSON `error`
 * message when available.
 */
export async function apiGet<T>(url: string): Promise<T>
{
	const res = await fetch(url)
	if (!res.ok)
	{
		let message = `Request failed (${res.status})`
		try
		{
			const data = await res.json()
			if (data && typeof data.error === 'string') message = data.error
		}
		catch
		{
			// Non-JSON error body — keep the status-based message
		}
		throw new Error(message)
	}
	return res.json() as Promise<T>
}

export interface LatestGuard
{
	/** Start a new request; returns its token and invalidates all older ones. */
	begin(): number
	/** True iff `token` belongs to the most recently begun request. */
	isCurrent(token: number): boolean
}

/**
 * Request-token guard: lets async loaders drop stale responses so a slow
 * old request never overwrites state from a newer one.
 *
 *   const guard = makeLatestGuard();
 *   async function load() {
 *     const token = guard.begin();
 *     const data = await apiGet(...);
 *     if (!guard.isCurrent(token)) return; // stale — drop
 *     state = data;
 *   }
 */
export function makeLatestGuard(): LatestGuard
{
	let latest = 0
	return {
		begin: () => ++latest,
		isCurrent: (token: number) => token === latest,
	}
}
