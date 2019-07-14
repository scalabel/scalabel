import { myTheme } from './theme'

// File for general styles which are used for multi-page components

export const defaultAppBar = {
  background: myTheme.palette.primary.dark,
  height: myTheme.mixins.toolbar.minHeight
}

export const defaultHeader = {
  ...myTheme.mixins.toolbar
}
