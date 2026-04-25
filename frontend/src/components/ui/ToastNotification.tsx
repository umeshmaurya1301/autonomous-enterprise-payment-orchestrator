"use client";
import { X, AlertTriangle, AlertCircle, Info } from "lucide-react";
import { cn } from "@/lib/utils";
import type { CausalNotification } from "@/lib/types";

interface ToastNotificationProps {
  notification: CausalNotification;
  onDismiss: (id: string) => void;
}

const SEVERITY_STYLES = {
  critical: {
    container: "border-red-500/50 bg-red-950/80 glow-red",
    icon: <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />,
    text: "text-red-300",
  },
  warning: {
    container: "border-yellow-500/50 bg-yellow-950/80 glow-yellow",
    icon: <AlertTriangle className="w-4 h-4 text-yellow-400 shrink-0" />,
    text: "text-yellow-300",
  },
  info: {
    container: "border-blue-500/50 bg-blue-950/80",
    icon: <Info className="w-4 h-4 text-blue-400 shrink-0" />,
    text: "text-blue-300",
  },
};

export function ToastNotification({ notification, onDismiss }: ToastNotificationProps) {
  const styles = SEVERITY_STYLES[notification.severity];

  return (
    <div
      className={cn(
        "flex items-center gap-3 px-4 py-3 rounded-lg border backdrop-blur-sm text-sm font-mono animate-slide-up",
        styles.container
      )}
    >
      {styles.icon}
      <span className={cn("flex-1 text-xs", styles.text)}>{notification.message}</span>
      <button
        onClick={() => onDismiss(notification.id)}
        className="text-slate-500 hover:text-slate-300 transition-colors"
      >
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

interface ToastContainerProps {
  notifications: CausalNotification[];
  onDismiss: (id: string) => void;
}

export function ToastContainer({ notifications, onDismiss }: ToastContainerProps) {
  if (notifications.length === 0) return null;

  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2 max-w-sm w-full">
      {notifications.map((n) => (
        <ToastNotification key={n.id} notification={n} onDismiss={onDismiss} />
      ))}
    </div>
  );
}
