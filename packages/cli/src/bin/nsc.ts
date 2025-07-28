#!/usr/bin/env node

/**
 * Neural Semantic Compiler - Node.js CLI Binary
 * 
 * Command-line interface for the Neural Semantic Compiler.
 */

import { Command } from 'commander';
import chalk from 'chalk';
import ora from 'ora';
import inquirer from 'inquirer';
import { readFileSync, writeFileSync } from 'fs';
import { NeuralSemanticCLI } from '../index';
import { CompressionLevel } from '@neurosemantic/types';

const program = new Command();

program
  .name('nsc')
  .description(' Neural Semantic Compiler - Node.js CLI')
  .version('1.0.0');

// Compress command
program
  .command('compress')
  .description('Compress text using Neural Semantic Compiler')
  .argument('[text]', 'Text to compress')
  .option('-i, --input <file>', 'Input file path')
  .option('-o, --output <file>', 'Output file path')
  .option('-l, --level <level>', 'Compression level (light, balanced, aggressive)', 'balanced')
  .option('-d, --domain <domain>', 'Domain context (web-dev, agile, etc.)')
  .option('-e, --engine <engine>', 'Specific engine to use (semantic, hybrid, extreme)')
  .option('--preserve-code', 'Preserve code blocks')
  .option('--preserve-urls', 'Preserve URLs')
  .option('--preserve-numbers', 'Preserve numbers')
  .option('--show-stats', 'Show compression statistics')
  .option('--json', 'Output as JSON')
  .option('--verbose', 'Verbose output')
  .action(async (text, options) => {
    try {
      const cli = new NeuralSemanticCLI({ verbose: options.verbose });
      
      // Get input text
      let inputText = text;
      if (options.input) {
        inputText = readFileSync(options.input, 'utf-8');
      } else if (!inputText) {
        const answer = await inquirer.prompt([
          {
            type: 'editor',
            name: 'text',
            message: 'Enter text to compress:',
            validate: (input) => input.trim().length > 0 || 'Text cannot be empty'
          }
        ]);
        inputText = answer.text;
      }

      // Show progress
      const spinner = ora('Compressing text...').start();

      // Perform compression
      const result = await cli.compress(inputText, {
        level: options.level as CompressionLevel,
        domain: options.domain,
        engine: options.engine,
        preserveCode: options.preserveCode,
        preserveUrls: options.preserveUrls,
        preserveNumbers: options.preserveNumbers
      });

      spinner.succeed('Compression completed');

      // Output result
      if (options.json) {
        const output = JSON.stringify(result, null, 2);
        if (options.output) {
          writeFileSync(options.output, output);
          console.log(chalk.green(`✓ Results saved to: ${options.output}`));
        } else {
          console.log(output);
        }
      } else {
        if (options.output) {
          writeFileSync(options.output, result.compressedText);
          console.log(chalk.green(`✓ Compressed text saved to: ${options.output}`));
        } else {
          console.log('\n' + chalk.blue.bold('Compressed Text:'));
          console.log(chalk.cyan(result.compressedText));
        }

        // Show statistics
        const savings = Math.round((1 - result.compressionRatio) * 100);
        const tokenSavings = result.originalTokens - result.compressedTokens;
        
        console.log(`\n${chalk.green('✓')} Compression: ${chalk.bold(savings + '%')} reduction`);
        console.log(chalk.dim(`Tokens: ${result.originalTokens} → ${result.compressedTokens} (saved ${tokenSavings})`));
        console.log(chalk.dim(`Quality: ${result.qualityScore.toFixed(1)}/10 | Engine: ${result.engineUsed}`));

        if (result.warnings.length > 0) {
          console.log('\n' + chalk.yellow('Warnings:'));
          result.warnings.forEach(warning => {
            console.log(`  ${chalk.yellow('⚠')} ${warning}`);
          });
        }

        if (options.showStats) {
          console.log('\n' + chalk.blue.bold('Detailed Statistics:'));
          console.log(`  Original length: ${inputText.length} chars`);
          console.log(`  Compressed length: ${result.compressedText.length} chars`);
          console.log(`  Character reduction: ${inputText.length - result.compressedText.length} chars`);
          console.log(`  Processing time: ${result.processingTimeMs}ms`);
          console.log(`  Pattern matches: ${result.patternMatches.length}`);
        }
      }

    } catch (error) {
      ora().fail(`Compression failed: ${(error as Error).message}`);
      process.exit(1);
    }
  });

// Patterns command
program
  .command('patterns')
  .description('Manage compression patterns')
  .option('-d, --domain <domain>', 'Filter by domain')
  .option('-t, --type <type>', 'Filter by pattern type')
  .option('-l, --limit <number>', 'Limit number of results', '20')
  .option('-s, --search <query>', 'Search patterns by text')
  .option('--json', 'Output as JSON')
  .action(async (options) => {
    try {
      const cli = new NeuralSemanticCLI();
      const patterns = await cli.getPatterns({
        domain: options.domain,
        patternType: options.type,
        limit: parseInt(options.limit),
        search: options.search
      });

      if (options.json) {
        console.log(JSON.stringify(patterns, null, 2));
      } else {
        if (patterns.length === 0) {
          console.log(chalk.yellow('No patterns found matching criteria'));
          return;
        }

        console.log(chalk.blue.bold(`\nCompression Patterns (${patterns.length} found):\n`));
        
        patterns.forEach((pattern, i) => {
          const original = pattern.original.length > 30 
            ? pattern.original.substring(0, 27) + '...'
            : pattern.original;
          
          console.log(`${i + 1}. ${chalk.cyan(original)} → ${chalk.green(pattern.compressed)}`);
          console.log(`   ${chalk.dim(`Type: ${pattern.patternType}, Domain: ${pattern.domain}, Priority: ${pattern.priority}`)}`);
        });
      }

    } catch (error) {
      console.error(chalk.red(`Error: ${(error as Error).message}`));
      process.exit(1);
    }
  });

// Add pattern command
program
  .command('add')
  .description('Add a new compression pattern')
  .argument('<original>', 'Original text')
  .argument('<compressed>', 'Compressed text')
  .option('-t, --type <type>', 'Pattern type', 'word')
  .option('-d, --domain <domain>', 'Domain', 'general')
  .option('-p, --priority <number>', 'Priority (100-1000)', '500')
  .option('--language <lang>', 'Language code', 'en')
  .action(async (original, compressed, options) => {
    try {
      const cli = new NeuralSemanticCLI();
      const success = await cli.addPattern(original, compressed, {
        patternType: options.type,
        domain: options.domain,
        priority: parseInt(options.priority),
        language: options.language
      });

      if (success) {
        console.log(chalk.green(`✓ Pattern added: ${chalk.cyan(original)} → ${chalk.green(compressed)}`));
      } else {
        console.log(chalk.red('✗ Failed to add pattern (conflict detected or already exists)'));
      }

    } catch (error) {
      console.error(chalk.red(`Error: ${(error as Error).message}`));
      process.exit(1);
    }
  });

// Compare command
program
  .command('compare')
  .description('Compare all compression engines')
  .argument('<text>', 'Text to compare')
  .option('-l, --level <level>', 'Compression level', 'balanced')
  .option('-d, --domain <domain>', 'Domain context')
  .option('--json', 'Output as JSON')
  .action(async (text, options) => {
    try {
      const cli = new NeuralSemanticCLI();
      const spinner = ora('Comparing engines...').start();
      
      const results = await cli.compareEngines(text, {
        level: options.level as CompressionLevel,
        domain: options.domain
      });

      spinner.succeed('Engine comparison completed');

      if (options.json) {
        console.log(JSON.stringify(results, null, 2));
      } else {
        console.log(chalk.blue.bold('\nEngine Comparison:\n'));
        
        Object.entries(results).forEach(([engine, result]) => {
          console.log(chalk.bold(engine.charAt(0).toUpperCase() + engine.slice(1)));
          
          if ('error' in result) {
            console.log(`  ${chalk.red('✗ Error:')} ${result.error}`);
          } else {
            const compression = Math.round((1 - (result.compressionRatio || 1)) * 100);
            console.log(`  ${chalk.green('✓')} Compression: ${compression}%`);
            console.log(`  Quality: ${(result.qualityScore || 0).toFixed(1)}/10`);
            console.log(`  Token savings: ${result.tokenSavings || 0}`);
            console.log(`  Time: ${result.processingTimeMs || 0}ms`);
          }
          console.log();
        });
      }

    } catch (error) {
      ora().fail(`Comparison failed: ${(error as Error).message}`);
      process.exit(1);
    }
  });

// Stats command
program
  .command('stats')
  .description('Show comprehensive statistics')
  .option('--json', 'Output as JSON')
  .action(async (options) => {
    try {
      const cli = new NeuralSemanticCLI();
      const stats = await cli.getStatistics();

      if (options.json) {
        console.log(JSON.stringify(stats, null, 2));
      } else {
        console.log(chalk.blue.bold('\n Neural Semantic Compiler Statistics\n'));
        
        // Session statistics
        console.log(chalk.green.bold('Session Statistics:'));
        console.log(`  Compressions: ${chalk.cyan(stats.session.compressions)}`);
        console.log(`  Average compression: ${chalk.cyan(Math.round(stats.session.averageCompression * 100) + '%')}`);
        console.log(`  Total tokens saved: ${chalk.cyan(stats.session.totalSavings.toLocaleString())}`);
        console.log(`  Duration: ${chalk.cyan(stats.session.durationMinutes.toFixed(1))} minutes`);
        
        // Pattern statistics
        console.log(chalk.green.bold('\nPattern Database:'));
        console.log(`  Total patterns: ${chalk.cyan(stats.patterns.totalPatterns)}`);
        console.log(`  Domains: ${chalk.cyan(Object.keys(stats.patterns.patternsByDomain).length)}`);
        console.log(`  Types: ${chalk.cyan(Object.keys(stats.patterns.patternsByType).length)}`);
        
        // Engine statistics
        console.log(chalk.green.bold('\nCompression Engines:'));
        console.log(`  Available engines: ${chalk.cyan(stats.engines.totalEngines)}`);
        console.log(`  Engines: ${chalk.cyan(stats.engines.availableEngines.join(', '))}`);
      }

    } catch (error) {
      console.error(chalk.red(`Error: ${(error as Error).message}`));
      process.exit(1);
    }
  });

// Health command
program
  .command('health')
  .description('Check system health')
  .action(async () => {
    try {
      const cli = new NeuralSemanticCLI();
      const health = await cli.getHealth();

      const statusColor = health.overallStatus === 'healthy' ? 'green' 
        : health.overallStatus === 'warning' ? 'yellow' : 'red';
      
      console.log(`\n${chalk.bold('System Status:')} ${chalk[statusColor](health.overallStatus.toUpperCase())}\n`);

      // Component status
      console.log(chalk.blue.bold('Component Health:'));
      
      Object.entries(health.components).forEach(([component, details]) => {
        const status = details.status;
        const emoji = status === 'healthy' ? '' 
          : status === 'warning' ? '⚠️' 
          : status === 'error' ? '' : '❓';
        
        console.log(`  ${emoji} ${component.replace('_', ' ')}: ${status}`);
      });

      // Issues
      if (health.issues.length > 0) {
        console.log(chalk.red.bold('\nIssues Found:'));
        health.issues.forEach(issue => {
          console.log(`  ${chalk.red('•')} ${issue}`);
        });
      }

    } catch (error) {
      console.error(chalk.red(`Health check failed: ${(error as Error).message}`));
      process.exit(1);
    }
  });

// Interactive mode
program
  .command('interactive')
  .alias('i')
  .description('Start interactive compression session')
  .action(async () => {
    console.log(chalk.blue.bold('\n Neural Semantic Compiler - Interactive Mode\n'));
    
    const cli = new NeuralSemanticCLI();
    
    while (true) {
      try {
        const answers = await inquirer.prompt([
          {
            type: 'editor',
            name: 'text',
            message: 'Enter text to compress (or empty to exit):',
          },
          {
            type: 'list',
            name: 'level',
            message: 'Choose compression level:',
            choices: ['light', 'balanced', 'aggressive'],
            default: 'balanced',
            when: (answers) => answers.text.trim().length > 0
          },
          {
            type: 'input',
            name: 'domain',
            message: 'Domain (optional):',
            when: (answers) => answers.text.trim().length > 0
          }
        ]);

        if (!answers.text || answers.text.trim().length === 0) {
          console.log(chalk.blue('Goodbye! '));
          break;
        }

        const spinner = ora('Compressing...').start();
        
        const result = await cli.compress(answers.text, {
          level: answers.level as CompressionLevel,
          domain: answers.domain || undefined
        });

        spinner.succeed('Compression completed');

        console.log(`\n${chalk.blue.bold('Result:')}`);
        console.log(chalk.cyan(result.compressedText));
        
        const savings = Math.round((1 - result.compressionRatio) * 100);
        console.log(`\n${chalk.green('✓')} ${savings}% compression | Quality: ${result.qualityScore.toFixed(1)}/10\n`);

      } catch (error) {
        console.error(chalk.red(`Error: ${(error as Error).message}\n`));
      }
    }
  });

program.parse();