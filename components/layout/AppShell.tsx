import { ReactNode } from "react";
import styles from "../../styles/ChatWorkspace.module.css";

type AppShellProps = {
  sidebarOpen: boolean;
  sidebar: ReactNode;
  chatPanel: ReactNode;
  sourceDrawer: ReactNode;
  showSources: boolean;
  onCloseSidebar: () => void;
  onCloseSources: () => void;
};

export default function AppShell({
  sidebarOpen,
  sidebar,
  chatPanel,
  sourceDrawer,
  showSources,
  onCloseSidebar,
  onCloseSources,
}: AppShellProps) {
  return (
    <main className={styles.workspace}>
      {sidebarOpen ? sidebar : null}
      {sidebarOpen ? (
        <button
          type="button"
          className={styles.sidebarBackdrop}
          onClick={onCloseSidebar}
          aria-label="Close sidebar"
        />
      ) : null}

      {chatPanel}

      {showSources ? (
        <button
          type="button"
          className={styles.drawerBackdrop}
          onClick={onCloseSources}
          aria-label="Close source drawer"
        />
      ) : null}

      {sourceDrawer}
    </main>
  );
}
