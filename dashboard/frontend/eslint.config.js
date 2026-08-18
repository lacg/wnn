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

const houseStyle = {
	// Allman. allowSingleLine:false so `if (x) { y }` is not exempted.
	'@stylistic/brace-style': ['error', 'allman', { allowSingleLine: false }],
	// Semicolons are optional in TS and therefore omitted. This rule operates on
	// script code only — CSS in <style> is parsed separately and left alone.
	'@stylistic/semi': ['error', 'never'],
	// A literal TAB per level. Display width is the editor's job (.editorconfig).
	'@stylistic/indent': ['error', 'tab']
}

export default [
	{
		ignores: ['.svelte-kit/**', 'build/**', 'node_modules/**', 'dist/**']
	},
	{
		files: ['**/*.ts', '**/*.js'],
		languageOptions: { parser: tsParser },
		plugins: { '@stylistic': stylistic },
		rules: houseStyle
	},
	{
		files: ['**/*.svelte'],
		languageOptions: {
			parser: svelteParser,
			parserOptions: { parser: tsParser }
		},
		plugins: { '@stylistic': stylistic },
		rules: houseStyle
	}
]
