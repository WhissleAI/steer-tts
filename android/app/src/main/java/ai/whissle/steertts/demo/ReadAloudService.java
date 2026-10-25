package ai.whissle.steertts.demo;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.Build;
import android.os.IBinder;

/**
 * Reads shared or selected text aloud as a foreground service (playback continues after the app that shared it is
 * gone), with a notification to stop it. A shared link is fetched and its article read (Article). A new request
 * replaces the one playing.
 */
public class ReadAloudService extends Service {
    static final String EXTRA_TEXT = "text", ACTION_STOP = "ai.whissle.steertts.demo.STOP_READING";
    // default importance (silent): a low-importance one lands in the collapsed silent section, its Stop out of sight
    private static final String CHANNEL = "reading";
    private static final int ID = 1;

    @Override public IBinder onBind(Intent i) { return null; }

    @Override public int onStartCommand(Intent i, int flags, int startId) {
        if (i == null || ACTION_STOP.equals(i.getAction()) || i.getStringExtra(EXTRA_TEXT) == null) {
            Announcer.get(this).stop();
            stopSelf();
            return START_NOT_STICKY;
        }
        String shared = i.getStringExtra(EXTRA_TEXT);
        show(Article.isLink(shared) ? "Opening the link…" : preview(shared), startId, true);
        new Thread(() -> {
            Article a = Article.resolve(shared);
            show(a.title != null ? a.title : preview(a.text), startId, false);
            Announcer.get(this).say(a.text, "read-aloud", true, () -> stopSelf(startId));
        }, "steertts-read-aloud").start();
        return START_NOT_STICKY;
    }

    private void show(String what, int startId, boolean first) {
        NotificationManager nm = getSystemService(NotificationManager.class);
        nm.deleteNotificationChannel("read_aloud");   // the earlier low-importance channel
        nm.createNotificationChannel(channel());
        PendingIntent stop = PendingIntent.getService(this, 1,
                new Intent(this, ReadAloudService.class).setAction(ACTION_STOP), PendingIntent.FLAG_IMMUTABLE);
        Notification n = new Notification.Builder(this, CHANNEL)
                .setSmallIcon(android.R.drawable.ic_media_play)
                .setContentTitle("Reading aloud")
                .setContentText(what)
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .addAction(new Notification.Action.Builder(null, "Stop", stop).build())
                .build();
        if (!first) nm.notify(ID, n);
        else if (Build.VERSION.SDK_INT >= 29) startForeground(ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK);
        else startForeground(ID, n);
    }

    private static NotificationChannel channel() {
        NotificationChannel c = new NotificationChannel(CHANNEL, "Read aloud", NotificationManager.IMPORTANCE_DEFAULT);
        c.setSound(null, null);
        c.enableVibration(false);
        return c;
    }

    private static String preview(String s) {
        s = s.replaceAll("\\s+", " ").trim();
        return s.length() <= 80 ? s : s.substring(0, 80) + "…";
    }
}
