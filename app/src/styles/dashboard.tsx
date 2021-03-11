import { StyleRules, Theme } from "@material-ui/core"
import createStyles from "@material-ui/core/styles/createStyles"
/* Dashboard window styles */
export const dashboardWindowStyles = (
  theme: Theme
): StyleRules<"row" | "root" | "linkButton" | "headerCell" | "bodyCell", {}> =>
  createStyles({
    root: {
      paddingLeft: theme.spacing(2),
      paddingRight: theme.spacing(2),
      paddingTop: theme.spacing(1)
    },
    row: {
      background: theme.palette.action.hover
    },
    linkButton: {
      fontSize: "0.8rem",
      color: theme.palette.secondary.main
    },
    headerCell: {
      fontWeight: "bold",
      // fontSize: "0.8rem",
      // color: theme.palette.primary.contrastText,
      background: theme.palette.background.default
    },
    bodyCell: {
      paddingTop: 0,
      paddingBottom: 0
      // color: theme.palette.primary.contrastText,
      // background: theme.palette.background.default
    }
  })
export const headerStyle = (theme: Theme): StyleRules<"grow" | "chip", {}> =>
  createStyles({
    grow: {
      flexGrow: 1
    },
    chip: {
      marginRight: theme.spacing(2),
      marginLeft: theme.spacing(1)
    }
  })
export const sidebarStyle = (
  theme: Theme
): StyleRules<
  "root" | "listRoot" | "listItem" | "coloredListItem" | "link",
  {}
> =>
  createStyles({
    root: {
      background: theme.palette.background.default
    },
    listRoot: {
      marginTop: theme.spacing(2),
      width: "90%",
      marginLeft: "5%"
    },
    listItem: {
      textAlign: "center",
      margin: 0,
      paddingTop: 2,
      paddingBottom: 2
    },
    coloredListItem: {
      backgroundColor: theme.palette.action.hover
    },
    link: {
      textAlign: "center",
      marginTop: theme.spacing(2),
      width: "90%",
      marginLeft: "5%"
    }
  })
export const listEntryStyle = (): StyleRules<
  "listTag" | "listEntry" | "listContainer",
  {}
> =>
  createStyles({
    listTag: {
      textAlign: "right",
      fontWeight: "bold"
    },
    listEntry: {
      textAlign: "left"
    },
    listContainer: {
      margin: 0
    }
  })
/* Styles for worker and admin dashboard */
export const dashboardStyles = (
  theme: Theme
): StyleRules<"adminRoot" | "workerRoot" | "labelText" | "appBarSpacer", {}> =>
  createStyles({
    adminRoot: {
      paddingLeft: theme.spacing(3),
      paddingRight: theme.spacing(3)
    },
    workerRoot: {
      flexGrow: 1,
      paddingLeft: theme.spacing(3),
      paddingRight: theme.spacing(3)
    },
    labelText: {
      marginTop: theme.spacing(2)
    },
    appBarSpacer: theme.mixins.toolbar
  })
/* Dashboard header style */
export const dashboardHeaderStyles = createStyles({
  title: {
    flexGrow: 1
  }
})

/* Table styles for the user dashboard */
export const tableStyles = (
  theme: Theme
): StyleRules<"row" | "root" | "headerCell", {}> =>
  createStyles({
    root: {},
    headerCell: {
      fontWeight: "bold",
      // fontSize: "0.8rem",
      color: theme.palette.primary.light
    },
    row: {
      background: theme.palette.action.hover
    }
  })
/* TableCellStyles */
export const tableCellStyles = (
  theme: Theme
): StyleRules<"head" | "body", {}> =>
  createStyles({
    head: {
      backgroundColor: theme.palette.background.default
    },
    body: {
      // fontSize: 16
    }
  })
