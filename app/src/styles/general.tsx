import { scalabelTheme } from "./theme"

// General styles which are used for in components across all pages

export const defaultAppBar = {
  background: scalabelTheme.palette.primary.dark,
  height: scalabelTheme.mixins.toolbar.minHeight
}

export const defaultHeader = {
  ...scalabelTheme.mixins.toolbar
}
