import { StyleRules, Theme } from "@material-ui/core/styles"
import createStyles from "@material-ui/core/styles/createStyles"

import { defaultAppBar, defaultHeader } from "./general"

// Styles used in the create and dashboard navigation page
export const drawerWidth = 240

export const headerPageStyle = (
  theme: Theme
): StyleRules<"root" | "appBar", {}> =>
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
): StyleRules<
  "content" | "drawer" | "drawerPaper" | "drawerHeader" | "appBarSpacer",
  {}
> =>
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
