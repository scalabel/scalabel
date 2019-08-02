import { myTheme } from './theme'

// general styles which are used for in components across all pages

export const defaultAppBar = {
  background: myTheme.palette.primary.dark,
  height: myTheme.mixins.toolbar.minHeight
}

export const defaultHeader = {
  ...myTheme.mixins.toolbar
}
