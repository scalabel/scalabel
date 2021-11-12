import { createStyles, StyleRules } from "@mui/styles"
import { Theme } from "@mui/material/styles"

import { defaultAppBar, defaultHeader } from "./general"

// Styles used in the create and dashboard navigation page
export const drawerWidth = 240

type headerPageStyleKey = "root" | "appBar"
type dividedPageStyleKey =
  | "content"
  | "drawer"
  | "drawerPaper"
  | "drawerHeader"
  | "appBarSpacer"

export const headerPageStyle = (
  theme: Theme
): StyleRules<{}, headerPageStyleKey> =>
  createStyles({
    root: {
      display: "flex",
      alignItems: "left"
    },
    appBar: {
      ...defaultAppBar,
      zIndex: theme.zIndex.drawer + 1
    }
  })

export const dividedPageStyle = (
  theme: Theme
): StyleRules<{}, dividedPageStyleKey> =>
  createStyles({
    drawer: {
      width: drawerWidth,
      flexShrink: 0
    },

    drawerPaper: {
      width: drawerWidth,
      background: theme.palette.background.paper
    },

    drawerHeader: {
      ...defaultHeader
    },

    content: {
      flexGrow: 1
    },

    appBarSpacer: theme.mixins.toolbar
  })
