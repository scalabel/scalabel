import createMuiTheme, { ThemeOptions } from '@material-ui/core/styles/createMuiTheme'
import { Palette, PaletteOptions } from '@material-ui/core/styles/createPalette'

declare module '@material-ui/core/styles/createMuiTheme' {
  interface Theme {
    /** palette of Theme */
    palette: Palette
  }

  interface ThemeOptions {
    /** palette of ThemeOptions */
    palette?: PaletteOptions
  }
}

/**
 * This is createMyTheme function
 * that overwrites the primary main color
 */
export default function createMyTheme (_options: ThemeOptions) {
  return createMuiTheme({
    palette: {
      primary: {
        main: '#616161'
      },
      secondary: {
        main: '#cde6df',
        light: '#e5fafc',
        dark: '#dc004e'
      }
    }
  })
}

export const myTheme = createMyTheme({})
