import AppBar from "@mui/material/AppBar"
import CssBaseline from "@mui/material/CssBaseline"
import { withStyles } from "@mui/styles"
import Toolbar from "@mui/material/Toolbar"
import React, { ReactNode } from "react"

import { headerPageStyle } from "../styles/navigation_page"

export interface ClassType {
  /** root class */
  root: string
  /** top bar on page */
  appBar: string
}

export interface HeaderPageProps {
  /** divided page classes */
  classes: ClassType
  /** header content */
  headerContent: ReactNode
  /** main page content */
  pageContent: ReactNode
}

/**
 * Renders a page split with a header
 *
 * @param props
 * @constructor
 */
function HeaderPage(props: HeaderPageProps): JSX.Element {
  const { classes, headerContent, pageContent } = props
  return (
    <div className={classes.root}>
      <CssBaseline />
      <AppBar
        // position="static"
        style={{ background: "#000000" }}
        className={classes.appBar}
      >
        <Toolbar variant="dense">{headerContent}</Toolbar>
      </AppBar>
      {pageContent}
    </div>
  )
}

/** export divided page */
export default withStyles(headerPageStyle, { withTheme: true })(HeaderPage)
