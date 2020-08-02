import Drawer from '@material-ui/core/Drawer'
import { withStyles } from '@material-ui/core/styles'
import React, { ReactNode } from 'react'
import { dividedPageStyle } from '../styles/navigation_page'
import HeaderPage from './header_page'

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
  const mainPageContent = (
          <React.Fragment>
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
              <div className={classes.appBarSpacer}/>
              {children.mainContent}
            </main>
          </React.Fragment>
  )
  return (
          <HeaderPage children={{
            headerContent: children.headerContent,
            pageContent: mainPageContent
          }}
          />
  )
}

/** export divided page */
export default withStyles(dividedPageStyle, { withTheme: true })(DividedPage)
