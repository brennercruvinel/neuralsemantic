/**
 * Logger Utility for Neural Semantic Compiler
 */

export class Logger {
  private level: LogLevel;
  private logFile?: string;
  
  constructor(level: string = 'info', logFile?: string) {
    this.level = this.parseLogLevel(level);
    this.logFile = logFile;
  }

  debug(message: string, meta?: any): void {
    if (this.level <= LogLevel.DEBUG) {
      this.log('DEBUG', message, meta);
    }
  }

  info(message: string, meta?: any): void {
    if (this.level <= LogLevel.INFO) {
      this.log('INFO', message, meta);
    }
  }

  warn(message: string, meta?: any): void {
    if (this.level <= LogLevel.WARN) {
      this.log('WARN', message, meta);
    }
  }

  error(message: string, meta?: any): void {
    if (this.level <= LogLevel.ERROR) {
      this.log('ERROR', message, meta);
    }
  }

  private log(level: string, message: string, meta?: any): void {
    const timestamp = new Date().toISOString();
    const logEntry = {
      timestamp,
      level,
      message,
      ...(meta && { meta })
    };

    // Console output
    const formattedMessage = this.formatMessage(logEntry);
    
    switch (level) {
      case 'ERROR':
        console.error(formattedMessage);
        break;
      case 'WARN':
        console.warn(formattedMessage);
        break;
      case 'DEBUG':
        console.debug(formattedMessage);
        break;
      default:
        console.log(formattedMessage);
    }

    // File output (if configured)
    if (this.logFile) {
      this.writeToFile(logEntry);
    }
  }

  private formatMessage(entry: any): string {
    let formatted = `[${entry.timestamp}] ${entry.level}: ${entry.message}`;
    
    if (entry.meta) {
      const metaStr = typeof entry.meta === 'object' 
        ? JSON.stringify(entry.meta, null, 2)
        : String(entry.meta);
      formatted += `\n${metaStr}`;
    }
    
    return formatted;
  }

  private async writeToFile(entry: any): Promise<void> {
    if (!this.logFile) return;

    try {
      const fs = await import('fs').then(m => m.promises);
      const logLine = JSON.stringify(entry) + '\n';
      await fs.appendFile(this.logFile, logLine);
    } catch (error) {
      // Fallback to console if file writing fails
      console.error('Failed to write to log file:', error);
    }
  }

  private parseLogLevel(level: string): LogLevel {
    switch (level.toLowerCase()) {
      case 'debug':
        return LogLevel.DEBUG;
      case 'info':
        return LogLevel.INFO;
      case 'warn':
        return LogLevel.WARN;
      case 'error':
        return LogLevel.ERROR;
      default:
        return LogLevel.INFO;
    }
  }
}

enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3
}