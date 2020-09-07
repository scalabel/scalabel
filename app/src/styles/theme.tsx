import { Mixins, MixinsOptions } from "@material-ui/core/styles/createMixins"
import createMuiTheme, { Theme } from "@material-ui/core/styles/createMuiTheme"
import { Palette, PaletteOptions } from "@material-ui/core/styles/createPalette"

const titleBarHeight = 48

declare module "@material-ui/core/styles/createMuiTheme" {
  interface Theme {
    /** Palette of Theme */
    palette: Palette

    /** Mixins of Theme */
    mixins: Mixins
  }

  interface ThemeOptions {
    /** Palette of ThemeOptions */
    palette?: PaletteOptions
    /** Mixins of ThemeOptions */
    mixins?: MixinsOptions
  }
}

/**
 * This is createMyTheme function
 * that overwrites the primary main color
 */
export default function createScalabelTheme(): Theme {
  return createMuiTheme({
    palette: {
      type: "dark",
      primary: {
        main: "#1976d2" // blue 700
      },
      secondary: {
        main: "#009688" // teal 500
      },
      divider: "rgba(255, 255, 255, 0.24)"
    },
    typography: {
      fontFamily: '"Helvetica Neue", "Roboto", "Arial", sans-serif',
      h6: {
        fontWeight: "bold"
      }
    },
    mixins: {
      toolbar: {
        minHeight: titleBarHeight
      }
    }
  })
}

export const scalabelTheme = createScalabelTheme()
