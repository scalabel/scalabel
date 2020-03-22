import { Mixins, MixinsOptions } from '@material-ui/core/styles/createMixins'
import createMuiTheme, { ThemeOptions } from '@material-ui/core/styles/createMuiTheme'
import { Palette, PaletteOptions } from '@material-ui/core/styles/createPalette'
import 'typeface-exo-2'
import 'typeface-lato'

const titleBarHeight = 48

declare module '@material-ui/core/styles/createMuiTheme' {

  interface Theme {
    /** palette of Theme */
    palette: Palette

    /** mixins of Theme */
    mixins: Mixins
  }

  interface ThemeOptions {
    /** palette of ThemeOptions */
    palette?: PaletteOptions
    /** mixins of ThemeOptions */
    mixins?: MixinsOptions

  }
}

/**
 * This is createMyTheme function
 * that overwrites the primary main color
 */
export default function createMyTheme (_options: ThemeOptions) {
  return createMuiTheme({
    typography: {
      fontFamily: [
        'Lato'
      ].join(','),
      body1: {
        fontWeight: 400,
        lineHeight: 1.5
      },
      h6: {
        fontFamily: 'Exo-2'
      }
    },
    palette: {
      primary: {
        main: '#616161',
        dark: '#333'
      },
      secondary: {
        main: '#cde6df',
        light: '#e5fafc',
        dark: '#dc004e'
      }
    },
    mixins: {
      toolbar: {
        minHeight: titleBarHeight
      }
    }
  })
}

export const myTheme = createMyTheme({})
