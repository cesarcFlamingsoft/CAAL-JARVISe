import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../controllers/connection_error_ctrl.dart';

/// Displays MCP connection errors as a banner.
class ConnectionErrorBanner extends StatelessWidget {
  const ConnectionErrorBanner({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<ConnectionErrorCtrl>(
      builder: (context, ctrl, _) {
        if (!ctrl.hasErrors) {
          return const SizedBox.shrink();
        }

        return SafeArea(
          minimum: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          child: Align(
            alignment: Alignment.topCenter,
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 500),
              child: Material(
                elevation: 4,
                borderRadius: BorderRadius.circular(12),
                color: Colors.orange.withValues(alpha: 0.9),
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.warning_amber_outlined, color: Colors.white),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              'Connection Error',
                              style: TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            const SizedBox(height: 4),
                            ...ctrl.errors.map(
                              (error) => Text(
                                error,
                                style: const TextStyle(color: Colors.white, fontSize: 12),
                              ),
                            ),
                          ],
                        ),
                      ),
                      IconButton(
                        onPressed: ctrl.clearErrors,
                        icon: const Icon(Icons.close, color: Colors.white),
                        tooltip: 'Dismiss',
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}
