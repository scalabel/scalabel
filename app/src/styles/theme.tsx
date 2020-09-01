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
export default function createMyTheme(): Theme {
  return createMuiTheme({
    palette: {
      primary: {
        main: "#616161",
        dark: "#333"
      },
      secondary: {
        main: "#cde6df",
        light: "#e5fafc",
        dark: "#dc004e"
      }
    },
    mixins: {
      toolbar: {
        minHeight: titleBarHeight
      }
    }
  })
}

export const myTheme = createMyTheme()
