import { Mixins, MixinsOptions } from "@mui/material/styles/createMixins"
import { createTheme, Theme } from "@mui/material/styles"
import { Palette, PaletteOptions } from "@mui/material/styles/createPalette"

const titleBarHeight = 48

declare module "@mui/material/styles" {
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
  return createTheme({
    palette: {
      mode: "dark",
      primary: {
        main: "#1976d2" // blue 700
      },
      secondary: {
        main: "#009688" // teal 500
      },
      divider: "rgba(255, 255, 255, 0.24)",
      action: {
        hover: "rgba(25, 118, 210, 0.16)",
        selected: "rgba(25, 118, 210, 0.32)",
        disabled: "rgba(25, 118, 210, 0.06)",
        focus: "rgba(25, 118, 210, 0.24)"
      }
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
    },
    components: {
      MuiCssBaseline: {
        styleOverrides: {
          "@global": {
            "*": {
              scrollbarWidth: "thin",
              scrollbarColor: "#bdbdbd #0000"
            },
            ".MuiToggleButton-root": {
              color: "rgba(255, 255, 255, 0.7) !important",
              "&.Mui-selected": {
                color: "rgba(255, 255, 255, 1) !important",
                background: "rgba(25, 118, 210, 0.32) !important"
              },
              "&:hover": {
                background: "rgba(25, 118, 210, 0.16)"
              },
              "&:active": {
                background: "rgba(25, 118, 210, 0.24) !important"
              }
            },
            "*::-webkit-scrollbar": {
              width: "5px"
            },
            /* Track */
            "*::-webkit-scrollbar-track": {
              background: "#0000"
            },
            /* Handle */
            "*::-webkit-scrollbar-thumb": {
              background: "#bdbdbd"
            },
            /* Handle on hover */
            "*::-webkit-scrollbar-thumb:hover": {
              background: "#757575"
            }
          }
        }
      }
    }
  })
}

export const scalabelTheme = createScalabelTheme()
