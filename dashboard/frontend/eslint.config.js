// House style for the dashboard frontend — see docs/code_style_migration_plan.md
//
// Prettier is deliberately NOT used. It has no brace-position option and never will,
// so it can only ever produce K&R. ESLint + @stylistic has an explicit "allman" option
// and auto-fixes it, and — unlike a bespoke semicolon strip — it does not touch
// <style> blocks at all, so CSS semicolons and braces are structurally out of scope.
//
// Apply with:  npx eslint --fix .

import stylistic from '@stylistic/eslint-plugin'
import tsParser from '@typescript-eslint/parser'
import svelteParser from 'svelte-eslint-parser'
import sveltePlugin from 'eslint-plugin-svelte'
// Registered for its RULE DEFINITIONS only — none of its rules are enabled here.
// Source files carry inline `eslint-disable-next-line @typescript-eslint/...`
// directives, and ESLint errors on a directive naming a rule it cannot resolve.
import tsPlugin from '@typescript-eslint/eslint-plugin'

const houseStyle = {
	// Allman. allowSingleLine:false so `if (x) { y }` is not exempted.
	'@stylistic/brace-style': ['error', 'allman', { allowSingleLine: false }],
	// Semicolons are optional in TS and therefore omitted. This rule operates on
	// script code only — CSS in <style> is parsed separately and left alone.
	'@stylistic/semi': ['error', 'never'],
	// `semi` covers STATEMENTS. The delimiter between interface/type members is a
	// separate, also-optional construct that it does not touch — without this,
	// `interface X { a: string; }` keeps a semicolon the house style rejects.
	'@stylistic/member-delimiter-style': ['error', {
		multiline: { delimiter: 'none' },
		singleline: { delimiter: 'comma', requireLast: false }
	}],
	// A literal TAB per level. Display width is the editor's job (.editorconfig).
	'@stylistic/indent': ['error', 'tab']
}

export default [
	{
		ignores: ['.svelte-kit/**', 'build/**', 'node_modules/**', 'dist/**']
	},
	{
		// This config enables STYLISTIC rules only, so every inline
		// `eslint-disable-next-line @typescript-eslint/...` in the source looks
		// "unused" to it — and --fix DELETES unused directives. That silently threw
		// away a real annotation on the first run. Off: those directives document a
		// deliberate escape hatch and must survive a formatting pass.
		linterOptions: { reportUnusedDisableDirectives: 'off' }
	},
	{
		files: ['**/*.ts', '**/*.js'],
		languageOptions: { parser: tsParser },
		plugins: { '@stylistic': stylistic, '@typescript-eslint': tsPlugin },
		rules: houseStyle
	},
	{
		files: ['**/*.svelte'],
		languageOptions: {
			parser: svelteParser,
			parserOptions: { parser: tsParser }
		},
		plugins: { '@stylistic': stylistic, svelte: sveltePlugin },
		rules: {
			...houseStyle,
			// @stylistic/indent does NOT reach inside a .svelte <script> block — it
			// leaves the original leading spaces and only tabs what it adds, so the
			// blocks come out space/tab mixed. svelte/indent is Svelte-aware and
			// indents markup and script together; it replaces the generic rule here.
			'@stylistic/indent': 'off',
			'svelte/indent': ['error', { indent: 'tab' }]
		}
	}
]
