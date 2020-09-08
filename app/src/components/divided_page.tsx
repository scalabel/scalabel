import Drawer from "@material-ui/core/Drawer"
import { withStyles } from "@material-ui/core/styles"
import React, { ReactNode } from "react"

import { dividedPageStyle } from "../styles/navigation_page"
import HeaderPage from "./header_page"

export interface ClassType {
  /** sidebar class */
  drawer: string
  /** sidebar background */
  drawerPaper: string
  /** sidebar header */
  drawerHeader: string
  /** class for main content */
  content: string
  /** app bar spacer class */
  appBarSpacer: string
}

export interface DividedPageProps {
  /** divided page classes */
  classes: ClassType
  /** divided page header content */
  header: ReactNode
  /** divided page sidebar content */
  sidebar: ReactNode
  /** divided page main content */
  main: ReactNode
}

/**
 * Renders a page divided into a header, a drawer, and the main content
 *
 * @param props
 * @constructor
 */
function DividedPage(props: DividedPageProps): JSX.Element {
  const { classes, header, sidebar, main } = props
  const mainPage = (
    <React.Fragment>
      <Drawer
        className={classes.drawer}
        variant="permanent"
        anchor="left"
        classes={{
          paper: classes.drawerPaper
        }}
      >
        <div className={classes.drawerHeader} />
        {sidebar}
      </Drawer>
      <main className={classes.content}>
        <div className={classes.appBarSpacer} />
        {main}
      </main>
    </React.Fragment>
  )
  return <HeaderPage headerContent={header} pageContent={mainPage} />
}

/** export divided page */
export default withStyles(dividedPageStyle, { withTheme: true })(DividedPage)
