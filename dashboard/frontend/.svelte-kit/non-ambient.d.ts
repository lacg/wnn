
// this file is generated — do not edit it


declare module "svelte/elements" {
	export interface HTMLAttributes<T> {
		'data-sveltekit-keepfocus'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-noscroll'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-preload-code'?:
			| true
			| ''
			| 'eager'
			| 'viewport'
			| 'hover'
			| 'tap'
			| 'off'
			| undefined
			| null;
		'data-sveltekit-preload-data'?: true | '' | 'hover' | 'tap' | 'off' | undefined | null;
		'data-sveltekit-reload'?: true | '' | 'off' | undefined | null;
		'data-sveltekit-replacestate'?: true | '' | 'off' | undefined | null;
	}
}

export {};


declare module "$app/types" {
	export interface AppTypes {
		RouteId(): "/" | "/checkpoints" | "/experiments" | "/flows" | "/flows/new" | "/flows/[id]";
		RouteParams(): {
			"/flows/[id]": { id: string }
		};
		LayoutParams(): {
			"/": { id?: string };
			"/checkpoints": Record<string, never>;
			"/experiments": Record<string, never>;
			"/flows": { id?: string };
			"/flows/new": Record<string, never>;
			"/flows/[id]": { id: string }
		};
		Pathname(): "/" | "/checkpoints" | "/checkpoints/" | "/experiments" | "/experiments/" | "/flows" | "/flows/" | "/flows/new" | "/flows/new/" | `/flows/${string}` & {} | `/flows/${string}/` & {};
		ResolvedPathname(): `${"" | `/${string}`}${ReturnType<AppTypes['Pathname']>}`;
		Asset(): string & {};
	}
}