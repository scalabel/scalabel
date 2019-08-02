import AppBar from '@material-ui/core/AppBar'
import CssBaseline from '@material-ui/core/CssBaseline'
import Drawer from '@material-ui/core/Drawer'
import { withStyles } from '@material-ui/core/styles'
import Toolbar from '@material-ui/core/Toolbar'
import React, { ReactNode } from 'react'
import { dividedPageStyle } from '../styles/navigation_page'

export interface ClassType {
  /** root class */
  root: string
  /** top bar on page */
  appBar: string
  /** sidebar class */
  drawer: string
  /** sidebar background */
  drawerPaper: string
  /** sidebar header */
  drawerHeader: string
  /** class for main content */
  content: string
}

export interface DividedPageProps {
  /** divided page classes */
  classes: ClassType
  /** divided page children */
  children: {
    /** divided page header content */
    headerContent: ReactNode
    /** divided page sidebar content */
    sidebarContent: ReactNode
    /** divided page main content */
    mainContent: ReactNode
  }
}

/**
 * Renders a page divided into a header, a drawer, and the main content
 * @param props
 * @constructor
 */
function DividedPage (props: DividedPageProps) {
  const { classes, children } = props
  return (
          <div className={classes.root}>
            <CssBaseline/>
            <AppBar
                    className={classes.appBar}
            >
              <Toolbar>
                {children.headerContent}
              </Toolbar>
            </AppBar>
            <Drawer
                    className={classes.drawer}
                    variant='permanent'
                    anchor='left'
                    classes={{
                      paper: classes.drawerPaper
                    }}
            >
              <div className={classes.drawerHeader}/>
              {children.sidebarContent}
            </Drawer>

            <main className={classes.content}>
              {children.mainContent}
            </main>
          </div>
  )
}

/** export Dashboard page */
export default withStyles(dividedPageStyle, { withTheme: true })(DividedPage)
