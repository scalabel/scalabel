import React from 'react';
import PropTypes from 'prop-types';
import CssBaseline from '@material-ui/core';
import SplitPane from 'react-split-pane';

type Props = {
  titleBar: Object,
  leftSidebar1: Object,
  leftSidebar2?: Object,
  center: Object,
  bottomBar?: Object,
  rightSidebar1?: Object,
  rightSidebar2?: Object,
}

/**
 * Layout of the labeling interface
 */
class LabelLayout extends React.Component<Props> {
  /**
  * Render function
  * @return {React.Fragment} React fragment
  */
  render() {
    const {titleBar} = this.props;
    const {leftSidebar1} = this.props;
    const {leftSidebar2} = this.props;
    const {bottomBar} = this.props;
    const {center} = this.props;
    const {rightSidebar1} = this.props;
    const {rightSidebar2} = this.props;
    return (
        <React.Fragment>
          <CssBaseline />
          {titleBar}
          <main>
            <SplitPane split='vertical' minSize={180}
                       defaultSize={200} maxSize={300}>
              {/* left sidebar */}
              {leftSidebar2 ?
                  <SplitPane split='horizontal' minSize={180}
                             defaultSize={200} maxSize={-180}>
                    {leftSidebar1}
                    {leftSidebar2}
                  </SplitPane>
                  : leftSidebar1
              }

              <SplitPane split='vertical' minSize={500}
                         defaultSize='80%' maxSize={-200}>
                {/* content */}
                {bottomBar ?
                    <SplitPane split='horizontal' minSize={180}
                               defaultSize='70%' maxSize={-180}>
                      {center}
                      {bottomBar}
                    </SplitPane>
                    : center
                }

                {/* right sidebar */}
                {rightSidebar2 ?
                    <SplitPane split='horizontal' minSize={180}
                               defaultSize={200} maxSize={-180}>
                      {rightSidebar1}
                      {rightSidebar2}
                    </SplitPane>
                    : rightSidebar1
                }
              </SplitPane>
            </SplitPane>
          </main>
          {/* End footer */}
        </React.Fragment>
    );
  }
}

LabelLayout.propTypes = {
  titleBar: PropTypes.object.isRequired,
  leftSidebar1: PropTypes.object.isRequired,
  leftSidebar2: PropTypes.object,
  center: PropTypes.object.isRequired,
  bottomBar: PropTypes.object,
  rightSidebar1: PropTypes.object,
  rightSidebar2: PropTypes.object,
};

export default LabelLayout;
